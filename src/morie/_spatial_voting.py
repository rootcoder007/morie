"""Spatial voting and scaling models backend.

Implements methods from Armstrong (2021) "Analyzing Spatial Models of Choice
and Judgment" (2nd ed., Chapman & Hall/CRC).

All functions are pure NumPy/SciPy -- no external R packages required.
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn._array_core import NDArray


def _am_stimuli(Z) -> list:
    """Aldrich-McKelvey closed form for the stimulus positions.

    Minimising sum_i ||X_i c_i - z||^2 over unit-length, centred z, with
    X_i = [1, z_i], leaves z' sum_i (I - P_i) z, so z is the eigenvector of
    sum_i (I - P_i) with the smallest eigenvalue orthogonal to the constant
    (which every projection P_i reproduces).  Complete, non-constant rows
    only, as basicspace::aldmck.  Standardised (sd with n - 1), first
    stimulus on the left.
    """
    rows = [[float(v) for v in r] for r in Z]
    q = len(rows[0]) if rows else 0
    if q < 2:
        raise ValueError("Aldrich-McKelvey scaling needs at least two stimuli.")
    M = [[0.0] * q for _ in range(q)]
    used = 0
    for z in rows:
        if any(math.isnan(v) for v in z):
            continue
        mz = sum(z) / q
        szz = sum((v - mz) ** 2 for v in z)
        if szz == 0.0:
            continue
        # projection onto span{1, z}: 1/q + (z - mz)(z - mz)' / szz
        for j in range(q):
            for k in range(q):
                M[j][k] += (1.0 if j == k else 0.0) - 1.0 / q - (z[j] - mz) * (z[k] - mz) / szz
        used += 1
    if used < 1:
        raise ValueError(
            "Aldrich-McKelvey scaling needs at least one respondent who places every "
            "stimulus and not all at the same point."
        )
    # C M C with C the centring projection
    rm = [sum(M[j]) / q for j in range(q)]
    cm = [sum(M[j][k] for j in range(q)) / q for k in range(q)]
    gm = sum(rm) / q
    CMC = [[M[j][k] - rm[j] - cm[k] + gm for k in range(q)] for j in range(q)]
    vals, vecs = np.linalg.eigh(np.asarray(CMC))
    vals = [float(v) for v in vals]
    V = [[float(vecs[j][k]) for k in range(q)] for j in range(q)]
    best = None
    for k in range(q):
        col = [V[j][k] for j in range(q)]
        if abs(sum(col)) > 1e-8 * math.sqrt(q):
            continue  # the constant direction
        if best is None or vals[k] < vals[best]:
            best = k
    v = [V[j][best] for j in range(q)]
    mv = sum(v) / q
    sd = math.sqrt(sum((t - mv) ** 2 for t in v) / (q - 1))
    v = [(t - mv) / sd for t in v]
    if v[0] > 0:
        v = [-t for t in v]
    return v


def _arrays(res: dict, keys) -> dict:
    """Matrix-valued results as arrays (``.shape``, ``[:, j]``), like the rest of the module."""
    for k in keys:
        if k in res and res[k] is not None:
            res[k] = np.asarray(res[k])
    return res


def aldrich_mckelvey(Z: NDArray, n_dims: int = 1) -> dict:
    """Aldrich-McKelvey scaling (Aldrich & McKelvey 1977; Eqs 2.1-2.3).

    The stimulus positions are the least-squares solution: each
    respondent's placements are mapped onto the common scale by the best
    affine transformation and zhat minimises the total squared error, a
    closed form (the eigenvector in :func:`_am_stimuli`) that equals
    basicspace::aldmck.  The respondent intercepts and slopes regress each
    respondent's placements on zhat (any respondent with two placements).

    :param Z: (n_respondents x n_stimuli) matrix of perceptual placements.
    :param n_dims: Number of latent dimensions (must be 1).
    :return: dict with zhat, alpha, beta, weights, iterations (None: closed
        form), converged.
    """
    if int(n_dims) != 1:
        raise NotImplementedError(
            "aldrich_mckelvey: this implementation recovers a single latent dimension; n_dims must be 1"
        )
    Z = np.asarray(Z, dtype=float)
    n_resp, n_stim = Z.shape
    zhat = np.asarray(_am_stimuli(Z.tolist()))
    mask = ~np.isnan(Z)
    alpha = np.zeros(n_resp)
    beta = np.zeros(n_resp)
    for i in range(n_resp):
        valid = mask[i]
        if valid.sum() < 2:
            alpha[i] = 0.0
            beta[i] = 1.0
            continue
        zi = Z[i, valid]
        zh = zhat[valid]
        A = np.column_stack([np.ones(valid.sum()), zh])
        params, _, _, _ = np.linalg.lstsq(A, zi, rcond=None)
        alpha[i] = params[0]
        beta[i] = params[1] if abs(params[1]) > 1e-10 else 1e-10
    weights = np.abs(beta)
    weights = weights / weights.sum() * n_resp
    return {
        "zhat": zhat,
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "iterations": None,
        "converged": True,
    }


def blackbox_scaling(X: NDArray, n_dims: int = 2, minscale: int = 8) -> dict:
    """Blackbox / Basic Space scaling (Poole 1998; Eqs 2.4-2.8).

    X_0 = Psi W' + J_n c' + E_0 fitted by least squares over the observed
    cells only: missing cells are filled with the current fit, the column
    means and leading singular vectors of the centred matrix recomputed,
    and the two steps repeated until the filled cells stop moving (one SVD
    when nothing is missing).  Psi = U D^(1/2), W = V D^(1/2), the scaling
    basicspace::blackbox reports.  Respondents with fewer than
    ``minscale`` responses (capped at the number of issues) are not
    scaled; their rows are NaN.

    :param X: (n x p) matrix of issue scale responses (NaN for missing).
    :param n_dims: Number of dimensions to extract.
    :param minscale: Minimum responses for a respondent to be scaled.
    :return: dict with ideal_points, stimuli_weights, eigenvalues, fit.
    """
    rows = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    n = len(rows)
    p = len(rows[0]) if rows else 0
    if p < 2:
        raise ValueError("Blackbox scaling needs at least two issues (columns).")
    need = min(int(minscale), p)
    keep = [i for i in range(n) if sum(not math.isnan(v) for v in rows[i]) >= need]
    if len(keep) <= int(n_dims):
        raise ValueError("Too few respondents answer at least `minscale` issues.")
    Xk = [rows[i] for i in keep]
    nk = len(Xk)
    obs = [[not math.isnan(v) for v in r] for r in Xk]
    colmean = []
    for j in range(p):
        vals = [Xk[i][j] for i in range(nk) if obs[i][j]]
        colmean.append(sum(vals) / len(vals) if vals else 0.0)
    M = [[Xk[i][j] if obs[i][j] else colmean[j] for j in range(p)] for i in range(nk)]
    q = min(int(n_dims), p - 1, nk - 1)
    complete = all(all(o) for o in obs)
    for _ in range(5000):
        means = [sum(M[i][j] for i in range(nk)) / nk for j in range(p)]
        C = [[M[i][j] - means[j] for j in range(p)] for i in range(nk)]
        U, sv, Vt = np.linalg.svd(np.asarray(C), full_matrices=False)
        U = [[float(U[i][k]) for k in range(q)] for i in range(nk)]
        d = [float(sv[k]) for k in range(q)]
        Vt = [[float(Vt[k][j]) for j in range(p)] for k in range(q)]
        if complete:
            break
        change = 0.0
        for i in range(nk):
            for j in range(p):
                if not obs[i][j]:
                    f = means[j] + sum(U[i][k] * d[k] * Vt[k][j] for k in range(q))
                    change = max(change, abs(M[i][j] - f))
                    M[i][j] = f
        if change < 1e-10:
            break
    Psi = [[float("nan")] * q for _ in range(n)]
    for r, i in enumerate(keep):
        Psi[i] = [U[r][k] * math.sqrt(d[k]) for k in range(q)]
    W = [[Vt[k][j] * math.sqrt(d[k]) for k in range(q)] for j in range(p)]
    total = sum(sum(c * c for c in row) for row in C)
    return {
        "ideal_points": np.asarray(Psi),
        "stimuli_weights": np.asarray(W),
        "eigenvalues": np.asarray([t * t for t in d]),
        "singular_values": np.asarray(d),
        "explained_variance": (sum(t * t for t in d) / total) if total > 0 else 0.0,
        "col_means": np.asarray(means),
        "n_dims": q,
    }


def optimal_classification(
    votes: NDArray,
    n_dims: int = 1,
    max_iter: int = 500,
    n_restarts: int = 10,
) -> dict:
    """Optimal Classification -- nonparametric scaling (Eqs 2.9-2.14).

    Finds legislator ideal points and cutting planes that minimize
    classification errors on roll call votes.

    :param votes: (n_legislators x n_votes) matrix. 1=Yea, 0=Nay, NaN=missing.
    :param n_dims: Number of dimensions.
    :param max_iter: Max iterations per restart.
    :param n_restarts: Number of random restarts.
    :return: dict with ideal_points, cutting_normals, PRE, APRE, errors.
    """
    votes = np.asarray(votes, dtype=float)
    n_leg, n_vote = votes.shape
    rng = np.random.default_rng(42)

    best_errors = np.inf
    best_result = None

    for _ in range(n_restarts):
        x = rng.standard_normal((n_leg, n_dims))

        for _ in range(max_iter):
            normals = np.zeros((n_vote, n_dims))
            for j in range(n_vote):
                valid = ~np.isnan(votes[:, j])
                yea = (votes[:, j] == 1) & valid
                nay = (votes[:, j] == 0) & valid
                if yea.sum() == 0 or nay.sum() == 0:
                    continue
                yea_mean = x[yea].mean(axis=0)
                nay_mean = x[nay].mean(axis=0)
                normal = yea_mean - nay_mean
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[j] = normal / norm

            x_new = np.zeros_like(x)
            for i in range(n_leg):
                valid = ~np.isnan(votes[i])
                if valid.sum() == 0:
                    continue
                direction = np.zeros(n_dims)
                for j in np.where(valid)[0]:
                    if votes[i, j] == 1:
                        direction += normals[j]
                    else:
                        direction -= normals[j]
                norm = np.linalg.norm(direction)
                x_new[i] = direction / norm if norm > 0 else x[i]
            x = x_new

        total_errors = 0
        null_errors = 0
        for j in range(n_vote):
            valid = ~np.isnan(votes[:, j])
            yea_count = (votes[:, j] == 1)[valid].sum()
            nay_count = (votes[:, j] == 0)[valid].sum()
            null_errors += min(yea_count, nay_count)

            proj = x[valid] @ normals[j]
            midpoint = (
                (x[votes[:, j] == 1].mean(axis=0) @ normals[j] + x[votes[:, j] == 0].mean(axis=0) @ normals[j]) / 2
                if yea_count > 0 and nay_count > 0
                else 0
            )
            predicted = (proj >= midpoint).astype(float)
            actual = votes[valid.nonzero()[0], j]
            total_errors += np.nansum(predicted != actual)

        if total_errors < best_errors:
            best_errors = total_errors
            pre = (null_errors - total_errors) / null_errors if null_errors > 0 else 0.0
            best_result = {
                "ideal_points": x.copy(),
                "cutting_normals": normals.copy(),
                "PRE": pre,
                "APRE": pre,
                "total_errors": int(total_errors),
                "null_errors": int(null_errors),
                "n_dims": n_dims,
            }

    return best_result


def double_centering(D: NDArray) -> NDArray:
    """Double-center a distance/dissimilarity matrix (Eqs 3.3-3.5).

    B = -0.5 * H * A * H where A = D^2, H = I - (1/n)*J.

    :param D: (n x n) distance matrix.
    :return: Double-centered matrix B.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    A = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * H @ A @ H


def classical_mds(
    D: NDArray,
    n_dims: int = 2,
) -> dict:
    """Classical (metric) Multidimensional Scaling (Eqs 3.1-3.10).

    Torgerson scaling via double-centering and eigendecomposition.

    :param D: (n x n) distance/dissimilarity matrix.
    :param n_dims: Number of dimensions to extract.
    :return: dict with coordinates, eigenvalues, stress, fit.
    """
    D = np.asarray(D, dtype=float)
    B = double_centering(D)

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos = eigenvalues[:n_dims]
    pos = np.maximum(pos, 0)
    coords = eigenvectors[:, :n_dims] @ np.diag(np.sqrt(pos))

    d_model = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            d_model[i, j] = np.linalg.norm(coords[i] - coords[j])

    valid = D > 0
    stress = 0.0
    if valid.sum() > 0:
        # v0.9.5.6+: Kruskal stress-1 normalises by sum(D^2), not by
        # sum(d_model^2) (which collapses to zero when the model is
        # underfit). Pre-v0.9.5.6 used d_model^2 denom.
        denom = (D[valid] ** 2).sum()
        stress = np.sqrt(((d_model[valid] - D[valid]) ** 2).sum() / denom) if denom > 0 else 0.0

    abs_eig = np.abs(eigenvalues)
    total = abs_eig.sum()
    fit = pos.sum() / total if total > 0 else 0.0

    return {
        "coordinates": coords,
        "eigenvalues": eigenvalues[:n_dims],
        "stress": stress,
        "fit": fit,
        "B_matrix": B,
    }


def smacof(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
    weights: NDArray | None = None,
    init: NDArray | None = None,
) -> dict:
    """SMACOF stress minimization (Eqs 3.13-3.17).

    Iterative majorization algorithm for MDS.

    :param D: (n x n) dissimilarity matrix.
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance on stress change.
    :param weights: (n x n) weight matrix (default: uniform).
    :param init: (n x n_dims) initial configuration.
    :return: dict with coordinates, stress, iterations.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    W = np.ones((n, n)) if weights is None else np.asarray(weights, dtype=float)
    np.fill_diagonal(W, 0)

    V = np.diag(W.sum(axis=1))
    V_inv = np.linalg.pinv(V)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims)) if init is None else np.asarray(init, dtype=float)

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    def compute_stress(X, d_X):
        return np.sum(W * (D - d_X) ** 2) / 2

    def compute_B(d_X):
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and d_X[i, j] > 1e-12:
                    B[i, j] = -W[i, j] * D[i, j] / d_X[i, j]
            B[i, i] = -B[i, :].sum() + B[i, i]
        return B

    d_X = compute_distances(X)
    stress = compute_stress(X, d_X)

    for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
        B_mat = compute_B(d_X)
        X_new = V_inv @ B_mat @ X

        d_X = compute_distances(X_new)
        stress_new = compute_stress(X_new, d_X)

        if abs(stress - stress_new) < tol:
            X = X_new
            stress = stress_new
            break

        X = X_new
        stress = stress_new

    return {
        "coordinates": X,
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def nonmetric_mds(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """Nonmetric MDS with ordinal constraints (Eq 3.18).

    Monotone regression to preserve rank order of dissimilarities.

    :param D: (n x n) dissimilarity matrix.
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with coordinates, stress, iterations.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims))

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    def isotonic_regression(y, w=None):
        n = len(y)
        if w is None:
            w = np.ones(n)
        result = y.copy()
        block_start = list(range(n))
        block_size = [1] * n
        block_val = list(y)
        block_w = list(w)

        i = 0
        while i < len(block_val) - 1:
            if block_val[i] > block_val[i + 1]:
                new_w = block_w[i] + block_w[i + 1]
                block_val[i] = (block_w[i] * block_val[i] + block_w[i + 1] * block_val[i + 1]) / new_w
                block_w[i] = new_w
                block_size[i] += block_size[i + 1]
                del block_val[i + 1], block_w[i + 1], block_size[i + 1], block_start[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        pos = 0
        for i in range(len(block_val)):
            for _j in range(block_size[i]):
                result[pos] = block_val[i]
                pos += 1
        return result

    mask = np.triu_indices(n, k=1)
    d_orig = D[mask]
    order = np.argsort(d_orig)

    for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
        d_X = compute_distances(X)
        d_current = d_X[mask]

        d_ordered = d_current[order]
        d_hat_ordered = isotonic_regression(d_ordered)
        d_hat = np.zeros_like(d_current)
        d_hat[order] = d_hat_ordered

        D_hat = np.zeros((n, n))
        D_hat[mask] = d_hat
        D_hat = D_hat + D_hat.T

        result = smacof(D_hat, n_dims=n_dims, max_iter=1, init=X)
        X_new = result["coordinates"]

        stress = (
            np.sqrt(np.sum((d_X[mask] - d_hat) ** 2) / np.sum(d_X[mask] ** 2)) if np.sum(d_X[mask] ** 2) > 0 else 0.0
        )

        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break
        X = X_new

    return {
        "coordinates": X,
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def mds_fit_stats(eigenvalues: NDArray) -> dict:
    """MDS fit statistics -- Mardia criterion (Eqs 3.9-3.10).

    :param eigenvalues: Array of eigenvalues from MDS decomposition.
    :return: dict with fit_1d, fit_2d, fit_nd for various dimensionalities.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    abs_eig = np.abs(eigenvalues)
    total = abs_eig.sum()
    if total == 0:
        return {"fit_by_dim": [], "cumulative_fit": []}

    pos = np.maximum(eigenvalues, 0)
    fit_by_dim = []
    cumulative = 0.0
    cumulative_fit = []
    for _i, ev in enumerate(pos):
        f = ev / total
        cumulative += f
        fit_by_dim.append(f)
        cumulative_fit.append(cumulative)

    return {
        "fit_by_dim": fit_by_dim,
        "cumulative_fit": cumulative_fit,
        "eigenvalues": eigenvalues.tolist(),
    }


def unfolding_stress(
    X_r: NDArray,
    X_s: NDArray,
    D: NDArray,
    weights: NDArray | None = None,
) -> float:
    """Compute unfolding stress (Eq 4.6).

    sigma = sum_i sum_j w_ij * (d_ij(X) - delta_ij)^2.
    """
    X_r = np.asarray(X_r, dtype=float)
    X_s = np.asarray(X_s, dtype=float)
    D = np.asarray(D, dtype=float)

    n_r, n_dims = X_r.shape
    n_s = X_s.shape[0]

    d_model = np.zeros((n_r, n_s))
    for i in range(n_r):
        for j in range(n_s):
            d_model[i, j] = np.linalg.norm(X_r[i] - X_s[j])

    W = np.ones_like(D) if weights is None else np.asarray(weights, dtype=float)

    mask = ~np.isnan(D)
    return float(np.sum(W[mask] * (d_model[mask] - D[mask]) ** 2))


def mlsmu6(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
) -> dict:
    """MLSMU6 alternating least-squares unfolding (Eqs 4.7-4.24).

    Multidimensional Least-Squares Metric Unfolding.

    :param D: (n_resp x n_stim) distance/rating matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum alternations.
    :param tol: Convergence tolerance.
    :param n_restarts: Random restarts.
    :return: dict with respondent_coords, stimulus_coords, stress.
    """
    D = np.asarray(D, dtype=float)
    n_r, n_s = D.shape
    rng = np.random.default_rng(42)

    best_stress = np.inf
    best_result = None

    for _restart in range(n_restarts):
        X_r = rng.standard_normal((n_r, n_dims))
        X_s = rng.standard_normal((n_s, n_dims))
        X_r -= X_r.mean(axis=0)
        X_s -= X_s.mean(axis=0)

        D_hat = D - D.mean(axis=1, keepdims=True)

        prev_stress = np.inf
        for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
            d_model = np.zeros((n_r, n_s))
            for i in range(n_r):
                for j in range(n_s):
                    d_model[i, j] = np.linalg.norm(X_r[i] - X_s[j])
            d_model = np.maximum(d_model, 1e-12)

            grad_r = np.zeros_like(X_r)
            for i in range(n_r):
                for j in range(n_s):
                    diff = X_r[i] - X_s[j]
                    grad_r[i] += 2 * (d_model[i, j] - D_hat[i, j]) * diff / d_model[i, j]
            grad_r /= n_s

            eig_r = np.linalg.eigvalsh(grad_r.T @ grad_r)
            gamma_r = 2.0 / (n_s * max(eig_r.max(), 1e-10))
            X_r -= gamma_r * grad_r
            X_r -= X_r.mean(axis=0)

            grad_s = np.zeros_like(X_s)
            for j in range(n_s):
                for i in range(n_r):
                    diff = X_s[j] - X_r[i]
                    grad_s[j] += 2 * (d_model[i, j] - D_hat[i, j]) * diff / d_model[i, j]
            grad_s /= n_r

            eig_s = np.linalg.eigvalsh(grad_s.T @ grad_s)
            gamma_s = 2.0 / (n_r * max(eig_s.max(), 1e-10))
            X_s -= gamma_s * grad_s
            X_s -= X_s.mean(axis=0)

            stress = unfolding_stress(X_r, X_s, D)
            if abs(prev_stress - stress) / max(prev_stress, 1e-12) < tol:
                break
            prev_stress = stress

        if stress < best_stress:
            best_stress = stress
            best_result = {
                "respondent_coords": X_r.copy(),
                "stimulus_coords": X_s.copy(),
                "stress": stress,
                "iterations": iteration + 1,
                "converged": iteration + 1 < max_iter,
            }

    return best_result


def smacof_unfolding(
    D: NDArray,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """SMACOF rectangular unfolding (Eqs 4.25-4.35).

    Majorization-based unfolding for respondent-stimulus distances.

    :param D: (n_resp x n_stim) dissimilarity matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with respondent_coords, stimulus_coords, stress.
    """
    D = np.asarray(D, dtype=float)
    n_r, n_s = D.shape
    n = n_r + n_s
    rng = np.random.default_rng(42)

    X_r = rng.standard_normal((n_r, n_dims))
    X_s = rng.standard_normal((n_s, n_dims))

    D_full = np.zeros((n, n))
    D_full[:n_r, n_r:] = D
    D_full[n_r:, :n_r] = D.T

    W = np.zeros((n, n))
    W[:n_r, n_r:] = 1.0
    W[n_r:, :n_r] = 1.0

    X = np.vstack([X_r, X_s])

    V = np.diag(W.sum(axis=1))
    V_inv = np.linalg.pinv(V)

    def compute_distances(X):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = np.linalg.norm(X[i] - X[j])
        return d

    d_X = compute_distances(X)
    stress = np.sum(W * (D_full - d_X) ** 2) / 2

    for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and d_X[i, j] > 1e-12:
                    B[i, j] = -W[i, j] * D_full[i, j] / d_X[i, j]
            B[i, i] = -B[i, :].sum() + B[i, i]

        X_new = V_inv @ B @ X

        d_X = compute_distances(X_new)
        stress_new = np.sum(W * (D_full - d_X) ** 2) / 2

        if abs(stress - stress_new) < tol:
            X = X_new
            stress = stress_new
            break

        X = X_new
        stress = stress_new

    return {
        "respondent_coords": X[:n_r],
        "stimulus_coords": X[n_r:],
        "stress": stress,
        "iterations": iteration + 1,
        "converged": iteration + 1 < max_iter,
    }


def ideal_point_recovery(
    X_r: NDArray,
    X_s: NDArray,
) -> NDArray:
    """Recover ideal points from unfolding configuration (Eq 4.36).

    ideal_point_i = x_ri (respondent position IS the ideal point).
    """
    return np.asarray(X_r, dtype=float).copy()


def nominate_utility(
    x: NDArray,
    z_yea: NDArray,
    z_nay: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> dict:
    """NOMINATE Gaussian utility model (Eqs 5.1-5.6).

    U_ijy = beta * exp(-0.5 * sum w_k^2 * d_ijky^2) + epsilon.

    :param x: (n_leg x n_dims) legislator ideal points.
    :param z_yea: (n_votes x n_dims) yea outcome locations.
    :param z_nay: (n_votes x n_dims) nay outcome locations.
    :param beta: Signal-to-noise ratio parameter.
    :param w: (n_dims,) dimension weights.
    :return: dict with utilities, vote_probs, utility_diff.
    """
    x = np.asarray(x, dtype=float)
    z_yea = np.asarray(z_yea, dtype=float)
    z_nay = np.asarray(z_nay, dtype=float)
    n_leg = x.shape[0]
    n_votes = z_yea.shape[0]
    n_dims = x.shape[1] if x.ndim > 1 else 1

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z_yea.ndim == 1:
        z_yea = z_yea.reshape(-1, 1)
    if z_nay.ndim == 1:
        z_nay = z_nay.reshape(-1, 1)

    w = np.ones(n_dims) if w is None else np.asarray(w, dtype=float)

    U_yea = np.zeros((n_leg, n_votes))
    U_nay = np.zeros((n_leg, n_votes))

    for i in range(n_leg):
        for j in range(n_votes):
            d_yea = np.sum(w**2 * (x[i] - z_yea[j]) ** 2)
            d_nay = np.sum(w**2 * (x[i] - z_nay[j]) ** 2)
            U_yea[i, j] = beta * np.exp(-0.5 * d_yea)
            U_nay[i, j] = beta * np.exp(-0.5 * d_nay)

    v = U_yea - U_nay
    P = 1.0 / (1.0 + np.exp(-v))

    return {
        "U_yea": U_yea,
        "U_nay": U_nay,
        "utility_diff": v,
        "vote_probs": P,
    }


def nominate_vote_prob(
    x_i: NDArray,
    z_yea_j: NDArray,
    z_nay_j: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> float:
    """Single vote probability under NOMINATE (Eqs 5.5-5.6)."""
    x_i = np.asarray(x_i, dtype=float).ravel()
    z_yea_j = np.asarray(z_yea_j, dtype=float).ravel()
    z_nay_j = np.asarray(z_nay_j, dtype=float).ravel()

    w = np.ones_like(x_i) if w is None else np.asarray(w, dtype=float)

    d_yea = np.sum(w**2 * (x_i - z_yea_j) ** 2)
    d_nay = np.sum(w**2 * (x_i - z_nay_j) ** 2)
    v = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
    return float(1.0 / (1.0 + np.exp(-v)))


def nominate_loglik(
    votes: NDArray,
    x: NDArray,
    z_yea: NDArray,
    z_nay: NDArray,
    beta: float = 15.0,
    w: NDArray | None = None,
) -> dict:
    """NOMINATE log-likelihood and GMP (Eqs 5.7-5.9).

    :param votes: (n_leg x n_votes) vote matrix. 1=Yea, 0=Nay, NaN=missing.
    :return: dict with loglik, GMP, n_correct, n_total.
    """
    result = nominate_utility(x, z_yea, z_nay, beta, w)
    P = result["vote_probs"]
    votes = np.asarray(votes, dtype=float)

    mask = ~np.isnan(votes)
    ll = 0.0
    n_correct = 0
    n_total = 0

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if not mask[i, j]:
                continue
            p = np.clip(P[i, j], 1e-10, 1 - 1e-10)
            if votes[i, j] == 1:
                ll += np.log(p)
                if p > 0.5:
                    n_correct += 1
            else:
                ll += np.log(1 - p)
                if p < 0.5:
                    n_correct += 1
            n_total += 1

    gmp = n_correct / n_total if n_total > 0 else 0.0

    return {
        "loglik": ll,
        "GMP": gmp,
        "n_correct": n_correct,
        "n_total": n_total,
    }


def procrustes_rotation(
    X: NDArray,
    X_target: NDArray,
) -> dict:
    """Procrustes rotation/alignment (Eqs 5.10-5.12).

    Find rotation T minimizing ||X_target - X @ T||_F.

    :param X: (n x p) configuration to rotate.
    :param X_target: (n x p) target configuration.
    :return: dict with rotated, rotation_matrix, scale, mse.
    """
    X = np.asarray(X, dtype=float)
    X_target = np.asarray(X_target, dtype=float)

    X_c = X - X.mean(axis=0)
    X_t = X_target - X_target.mean(axis=0)

    M = X_c.T @ X_t
    U, S, Vt = np.linalg.svd(M)
    T = U @ Vt

    if np.linalg.det(T) < 0:
        Vt[-1, :] *= -1
        T = U @ Vt

    X_rotated = X_c @ T + X_target.mean(axis=0)

    mse = float(np.mean((X_rotated - X_target) ** 2))

    return {
        "rotated": X_rotated,
        "rotation_matrix": T,
        "scale": float(S.sum()),
        "mse": mse,
    }


def bayesian_am_scaling(
    Z: NDArray,
    n_samples: int = 1000,
    burn_in: int = 200,
    polarity: int = 0,
    seed: int = 42,
) -> dict:
    """Bayesian Aldrich-McKelvey scaling (Hare et al. 2015).

    z_ij ~ N(a_i + b_i zhat_j, 1 / (tau_i tau_j)) with the priors of the
    authors' JAGS model: a_i, b_i ~ U(-100, 100), tau_j ~ G(0.1, 0.1),
    tau_i ~ G(g_a, g_b), g_a, g_b ~ G(0.1, 0.1), and zhat the standardised
    zstar ~ N(0, 1), the ``polarity`` stimulus (0-based) on the left.  Gibbs
    with truncated-normal a | b and b | a, conjugate gammas, and slice steps
    for g_a and each zstar_j; the posterior means and SDs agree with the
    JAGS model to the third decimal.  See :mod:`morie._bayes_scaling`.

    :return: dict with zeta_mean, zeta_sd, zeta_interval, a, b,
        tau_stimulus, draws, n_samples, engine.
    """
    from morie._bayes_scaling import bayes_am

    return _arrays(
        bayes_am(
            np.asarray(Z, dtype=float).tolist(),
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            polarity=int(polarity),
            seed=seed,
        ),
        ["zeta_mean", "zeta_sd", "zeta_interval", "a", "b", "tau_stimulus", "draws"],
    )


def bayesian_mds(
    D: NDArray,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
    sigma_init: float = 1.0,
    seed: int = 42,
) -> dict:
    """Bayesian metric MDS (Bakker & Poole 2013).

    log delta_ij ~ N(log ||x_i - x_j||, 1 / tau), x ~ N(0, 10^2),
    tau ~ U(0, 10) (so sigma >= 0.316), the model of the authors' JAGS
    code.  Coordinates slice sampled, tau drawn exactly; draws aligned
    (translation and rotation) on the posterior mean.

    :return: dict with positions, positions_sd, distance_mean, sigma, tau,
        draws, n_samples, engine.
    """
    from morie._bayes_scaling import bayes_mds

    return _arrays(
        bayes_mds(
            np.asarray(D, dtype=float).tolist(),
            n_dims=int(n_dims),
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            sigma_init=sigma_init,
            seed=seed,
        ),
        ["positions", "positions_sd", "distance_mean"],
    )


def bayesian_unfolding(
    D: NDArray,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
    seed: int = 42,
) -> dict:
    """Bayesian unfolding (Bakker & Poole 2013), the lognormal model on a
    respondent-by-stimulus dissimilarity matrix; feeling thermometers enter
    as (100 - T) / 50.

    :return: dict with stimuli, stimuli_sd, ideal_points, distance_mean,
        sigma, tau, n_samples, engine.
    """
    from morie._bayes_scaling import bayes_unfold

    return _arrays(
        bayes_unfold(
            np.asarray(D, dtype=float).tolist(),
            n_dims=int(n_dims),
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            seed=seed,
        ),
        ["stimuli", "stimuli_sd", "ideal_points", "distance_mean"],
    )


def cjr_irt(
    votes,
    n_dims: int = 1,
    n_samples: int = 1000,
    burn_in: int = 200,
    *,
    beta_prior_var: float = 25.0,
    start=None,
    seed: int = 0,
    keep_chains: bool = True,
) -> dict:
    """Clinton-Jackman-Rivers Bayesian IRT by Gibbs sampling (Eqs 6.17-6.26).

    ``P(y_ij = 1) = Phi(beta_j' x_i - alpha_j)`` with priors
    ``x_i ~ N(0, I)`` and ``(alpha_j, beta_j) ~ N(0, beta_prior_var I)``.
    Each sweep draws the latent utilities from their truncated normals
    (Albert and Chib 1993), then every ``(alpha_j, beta_j)`` and every
    ``x_i`` from its Gaussian full conditional, as in
    ``MCMCpack::MCMCirt1d`` and ``pscl::ideal``. All draws are inverse-CDF
    transforms of Philox uniforms (sweep ``t`` uses stream ``t``), so the
    Python and R arms give the same chain.

    :param votes: (n_leg x n_votes) matrix of 1, 0 and NaN (missing).
    :param n_dims: Number of ideal point dimensions.
    :param n_samples: Retained sweeps.
    :param burn_in: Discarded sweeps.
    :param beta_prior_var: Prior variance of alpha_j and beta_j.
    :param start: Optional (n_leg x n_dims) starting ideal points; default
        the ``em_irt`` posterior mode.
    :param seed: Philox key.
    :param keep_chains: Return the draws as well as the summaries.
    :return: dict with ideal_point_mean, ideal_point_sd, alpha_mean,
        beta_mean and, when kept, ideal_point_chain, alpha_chain, beta_chain.
    """
    from morie.fn._rng import normal_quantile, random_uniform

    Y = [[float(v) for v in row] for row in votes]
    N, J, D = len(Y), len(Y[0]), int(n_dims)
    obs = [[v == v for v in row] for row in Y]
    x = [list(map(float, r)) for r in start] if start is not None else em_irt(Y, n_dims=D)["ideal_points"]
    alpha, beta = [0.0] * J, [[0.0] * D for _ in range(J)]
    xs, as_, bs = [], [], []
    for t in range(burn_in + n_samples):
        u = [float(v) for v in random_uniform(N * J + J * (D + 1) + N * D, seed=seed, stream=t)]
        M = [[_em_dot(beta[j], x[i]) - alpha[j] for j in range(J)] for i in range(N)]
        q, far = [], {}
        for i in range(N):
            for j in range(J):
                v = u[i * J + j]
                s = M[i][j] if Y[i][j] == 1 else -M[i][j]
                if obs[i][j] and s < -30.0:
                    far[i * J + j] = _cjr_log_tail_quantile(math.log(v) + _cjr_log_phi_far(s))
                    v = 0.5
                elif obs[i][j]:
                    v *= _em_phi_cdf(s)
                q.append(v)
        q = [far.get(k, float(v)) for k, v in enumerate(normal_quantile(q))]
        ys = [
            [M[i][j] + (-q[i * J + j] if obs[i][j] and Y[i][j] == 1 else q[i * J + j]) for j in range(J)]
            for i in range(N)
        ]
        z_all = iter(float(v) for v in normal_quantile(u[N * J :]))
        xt = [[-1.0] + x[i] for i in range(N)]
        P = [
            [(1.0 / beta_prior_var if r == c else 0.0) + _em_col(xt, r, c) for c in range(D + 1)] for r in range(D + 1)
        ]
        V = _em_inverse(P)
        L = _cjr_chol(V)
        for j in range(J):
            rhs = [_em_ssum(xt[i][r] * ys[i][j] for i in range(N)) for r in range(D + 1)]
            mu = [_em_dot(V[r], rhs) for r in range(D + 1)]
            z = [next(z_all) for _ in range(D + 1)]
            draw = [mu[r] + _em_dot(L[r][: r + 1], z[: r + 1]) for r in range(D + 1)]
            alpha[j], beta[j] = draw[0], draw[1:]
        Q = [[(1.0 if r == c else 0.0) + _em_col(beta, r, c) for c in range(D)] for r in range(D)]
        W = _em_inverse(Q)
        K = _cjr_chol(W)
        for i in range(N):
            rhs = [_em_ssum(beta[j][d] * (ys[i][j] + alpha[j]) for j in range(J)) for d in range(D)]
            mu = [_em_dot(W[r], rhs) for r in range(D)]
            z = [next(z_all) for _ in range(D)]
            x[i] = [mu[r] + _em_dot(K[r][: r + 1], z[: r + 1]) for r in range(D)]
        if t >= burn_in:
            xs.append([list(r) for r in x])
            as_.append(list(alpha))
            bs.append([list(r) for r in beta])
    S = len(xs)

    def mean(ch, shape):
        return [[_em_ssum(c[i][d] for c in ch) / S for d in range(shape[1])] for i in range(shape[0])]

    xm = mean(xs, (N, D))
    out = {
        "ideal_point_mean": xm,
        "ideal_point_sd": [
            [math.sqrt(_em_ssum((c[i][d] - xm[i][d]) ** 2 for c in xs) / (S - 1)) if S > 1 else 0.0 for d in range(D)]
            for i in range(N)
        ],
        "alpha_mean": [_em_ssum(c[j] for c in as_) / S for j in range(J)],
        "beta_mean": mean(bs, (J, D)),
        "n_samples": n_samples,
    }
    if keep_chains:
        out.update(ideal_point_chain=xs, alpha_chain=as_, beta_chain=bs)
    return out


def _cjr_log_phi_far(m):
    # log Phi(m) for m < -30 from the Mills-ratio series; Phi itself underflows
    w = 1.0 / (m * m)
    return (
        -0.5 * m * m
        - math.log(-m)
        - 0.5 * math.log(2.0 * math.pi)
        + math.log(1.0 - w + 3.0 * w * w - 15.0 * w**3 + 105.0 * w**4)
    )


def _cjr_log_tail_quantile(logp):
    # AS 241 far-tail branch (r > 5) evaluated from log p
    from morie.fn._rng import _E, _F

    rr = math.sqrt(-logp) - 5.0
    num, den = _E[-1], _F[-1]
    for c in _E[-2::-1]:
        num = num * rr + c
    for c in _F[-2::-1]:
        den = den * rr + c
    return -(num / den)


def _cjr_chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(r + 1):
            s = A[r][c] - _em_ssum(L[r][k] * L[c][k] for k in range(c))
            L[r][c] = math.sqrt(s) if r == c else s / L[c][c]
    return L


def bayesian_irt_likelihood(
    votes: NDArray,
    x: NDArray,
    alpha: NDArray,
    beta: NDArray,
) -> dict:
    """Bayesian IRT likelihood computation (Eqs 6.27-6.28).

    L = prod_i prod_j P_ij^{y_ij} (1-P_ij)^{1-y_ij}.

    :param votes: (n_leg x n_votes) binary matrix.
    :param x: (n_leg x n_dims) ideal points.
    :param alpha: (n_votes,) difficulty parameters.
    :param beta: (n_votes x n_dims) discrimination parameters.
    :return: dict with loglik, vote_probs, n_correct.
    """
    from morie.fn._stats_core import norm

    votes = np.asarray(votes, dtype=float)
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)

    mask = ~np.isnan(votes)
    n_leg, n_vote = votes.shape

    P = np.zeros((n_leg, n_vote))
    ll = 0.0
    n_correct = 0
    n_total = 0

    for i in range(n_leg):
        for j in range(n_vote):
            if not mask[i, j]:
                continue
            z = float(beta[j] @ x[i] - alpha[j])
            p = np.clip(norm.cdf(z), 1e-10, 1 - 1e-10)
            P[i, j] = p
            if votes[i, j] == 1:
                ll += np.log(p)
                if p > 0.5:
                    n_correct += 1
            else:
                ll += np.log(1 - p)
                if p < 0.5:
                    n_correct += 1
            n_total += 1

    return {
        "loglik": ll,
        "vote_probs": P,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
    }


def bayesian_irt_posterior(
    chain: NDArray,
    standardize: bool = True,
) -> dict:
    """Posterior summaries and normalization for Bayesian IRT (Eqs 6.23-6.30).

    :param chain: (n_samples x n_leg x n_dims) MCMC chain of ideal points.
    :param standardize: Whether to standardize posteriors (Eq 6.29).
    :return: dict with posterior_mean, posterior_sd, credible_intervals, DIC.
    """
    chain = np.asarray(chain, dtype=float)
    n_samples = chain.shape[0]

    if standardize:
        for t in range(n_samples):
            m = chain[t].mean(axis=0)
            s = chain[t].std(axis=0)
            s = np.where(s > 0, s, 1.0)
            chain[t] = (chain[t] - m) / s

    posterior_mean = chain.mean(axis=0)
    posterior_sd = chain.std(axis=0)

    ci_low = np.percentile(chain, 2.5, axis=0)
    ci_high = np.percentile(chain, 97.5, axis=0)

    return {
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_samples": n_samples,
        "standardized": standardize,
    }


def ordered_optimal_classification(
    Y: NDArray,
    n_dims: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict:
    """Ordered Optimal Classification for ordinal issue scales (Section 2.4).

    Extends OC to ordinal response data by finding cutting hyperplanes
    that separate ordered categories nonparametrically.

    :param Y: (n_respondents x n_items) ordinal response matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with ideal_points, cutpoints, correct_class, iterations.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    mask = ~np.isnan(Y)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims))

    categories = {}
    for j in range(m):
        cats = np.unique(Y[mask[:, j], j])
        categories[j] = sorted(cats)

    cutpoints = {}
    for j in range(m):
        cats = categories[j]
        n_cuts = len(cats) - 1
        cutpoints[j] = np.linspace(-1, 1, n_cuts) if n_cuts > 0 else np.array([0.0])

    normals = rng.standard_normal((m, n_dims))
    for j in range(m):
        normals[j] /= np.linalg.norm(normals[j]) + 1e-12

    correct = 0
    total = 0
    for iteration in range(max_iter):
        old_correct = correct
        correct = 0
        total = 0

        for j in range(m):
            cats = categories[j]
            valid = mask[:, j]
            if len(cats) < 2:
                continue
            proj = X[valid] @ normals[j]
            y_valid = Y[valid, j]
            cuts = cutpoints[j]

            sorted_cuts = np.sort(cuts)
            predicted = np.digitize(proj, sorted_cuts)
            cat_map = {c: idx for idx, c in enumerate(cats)}
            actual = np.array([cat_map.get(v, 0) for v in y_valid])
            correct += np.sum(predicted == actual)
            total += len(actual)

        for j in range(m):
            cats = categories[j]
            valid = mask[:, j]
            if len(cats) < 2:
                continue
            proj = X[valid] @ normals[j]
            y_valid = Y[valid, j]
            cat_map = {c: idx for idx, c in enumerate(cats)}
            actual = np.array([cat_map.get(v, 0) for v in y_valid])
            sorted_idx = np.argsort(proj)
            new_cuts = []
            for k in range(len(cats) - 1):
                boundary_vals = []
                for ii in range(len(sorted_idx) - 1):
                    if actual[sorted_idx[ii]] <= k and actual[sorted_idx[ii + 1]] > k:
                        boundary_vals.append((proj[sorted_idx[ii]] + proj[sorted_idx[ii + 1]]) / 2)
                if boundary_vals:
                    new_cuts.append(np.median(boundary_vals))
                else:
                    new_cuts.append(cutpoints[j][k] if k < len(cutpoints[j]) else 0.0)
            cutpoints[j] = np.array(new_cuts)

        if total > 0 and iteration > 0 and abs(correct - old_correct) / total <= tol:
            break

    correct_rate = correct / total if total > 0 else 0.0
    return {
        "ideal_points": X,
        "cutpoints": cutpoints,
        "normals": normals,
        "correct_class": correct_rate,
        "iterations": iteration + 1,
    }


def anchoring_vignettes(
    Y: NDArray,
    V: NDArray,
    n_categories: int = 5,
) -> dict:
    """Anchoring vignettes for DIF correction (Section 2.5).

    Uses hypothetical vignette ratings to correct for differential item
    functioning (DIF) across respondent groups, enabling cross-group
    comparability of survey responses.

    :param Y: (n_respondents,) self-placement ratings.
    :param V: (n_respondents x n_vignettes) vignette ratings.
    :param n_categories: Number of ordered response categories.
    :return: dict with corrected_scores, thresholds, dif_estimates.
    """
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    n_resp = len(Y)
    n_vign = V.shape[1]

    vign_means = np.nanmean(V, axis=0)
    vign_order = np.argsort(vign_means)

    thresholds = np.zeros((n_resp, n_categories - 1))
    for i in range(n_resp):
        vi = V[i]
        valid = ~np.isnan(vi)
        if valid.sum() >= 2:
            sorted_v = np.sort(vi[valid])
            n_v = len(sorted_v)
            for k in range(n_categories - 1):
                idx = int(k * n_v / (n_categories - 1))
                idx = min(idx, n_v - 1)
                thresholds[i, k] = sorted_v[idx]
        else:
            thresholds[i] = np.linspace(1, n_categories, n_categories - 1)

    corrected = np.zeros(n_resp)
    for i in range(n_resp):
        y = Y[i]
        if np.isnan(y):
            corrected[i] = np.nan
            continue
        corrected[i] = np.searchsorted(thresholds[i], y)

    dif_estimates = np.std(thresholds, axis=0)

    return {
        "corrected_scores": corrected,
        "thresholds": thresholds,
        "dif_estimates": dif_estimates,
        "vignette_order": vign_order,
        "n_respondents": n_resp,
        "n_vignettes": n_vign,
    }


def indscal(
    dissimilarities: list[NDArray],
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """INDSCAL: Individual Differences MDS (Section 3.3, Eq 3.31).

    Carroll and Chang (1970) weighted MDS. Each individual has personal
    dimension weights applied to a common stimulus configuration.

    :param dissimilarities: List of (n_stim x n_stim) dissimilarity matrices per individual.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum ALS iterations.
    :param tol: Convergence tolerance.
    :return: dict with group_config, weights, stress, iterations.
    """
    n_indiv = len(dissimilarities)
    n_stim = dissimilarities[0].shape[0]

    D_list = [np.asarray(d, dtype=float) for d in dissimilarities]

    D_avg = np.mean(D_list, axis=0)
    n = D_avg.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_avg**2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_dims]
    X = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))

    W = np.ones((n_indiv, n_dims))

    for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
        X_old = X.copy()

        for k in range(n_indiv):
            D_k = D_list[k]
            for s in range(n_dims):
                X_s = X[:, s : s + 1]
                dist_s = np.sqrt(np.sum((X_s[:, None] - X_s[None, :]) ** 2, axis=-1) + 1e-12)
                numer = np.sum(D_k * dist_s)
                denom = np.sum(dist_s**2) + 1e-12
                W[k, s] = max(numer / denom, 0.01)

        for j in range(n_stim):
            for s in range(n_dims):
                numer = 0.0
                denom = 0.0
                for k in range(n_indiv):
                    for stim in range(n_stim):
                        if stim == j:
                            continue
                        d_kj = D_list[k][j, stim]
                        w_s = W[k, s]
                        numer += w_s * d_kj * X[stim, s]
                        denom += w_s**2 + 1e-12
                if denom > 0:
                    X[j, s] = numer / denom

        change = np.linalg.norm(X - X_old) / (np.linalg.norm(X_old) + 1e-12)
        if change < tol:
            break

    total_stress = 0.0
    for k in range(n_indiv):
        X_w = X * np.sqrt(W[k])
        d_model = np.sqrt(np.sum((X_w[:, None] - X_w[None, :]) ** 2, axis=-1) + 1e-12)
        total_stress += np.sum((D_list[k] - d_model) ** 2)

    return {
        "group_config": X,
        "weights": W,
        "stress": total_stress,
        "iterations": iteration + 1,
        "n_individuals": n_indiv,
        "n_stimuli": n_stim,
    }


def normal_vectors(
    ideal_points: NDArray,
    external_measure: NDArray,
) -> dict:
    """Normal vector projection onto recovered space (Eqs 2.12-2.13).

    Projects external measures (e.g., party ID, issue positions) onto the
    recovered latent space to produce directional normal vectors.

    :param ideal_points: (n x n_dims) recovered ideal point coordinates.
    :param external_measure: (n,) external variable to project.
    :return: dict with normal_vector, angle, r_squared, coefficients.
    """
    X = np.asarray(ideal_points, dtype=float)
    y = np.asarray(external_measure, dtype=float)

    mask = ~np.isnan(y)
    X_valid = X[mask]
    y_valid = y[mask]

    X_aug = np.column_stack([np.ones(X_valid.shape[0]), X_valid])
    beta, residuals, _, _ = np.linalg.lstsq(X_aug, y_valid, rcond=None)

    coeffs = beta[1:]
    norm = np.linalg.norm(coeffs)
    nv = coeffs / norm if norm > 0 else coeffs

    y_pred = X_aug @ beta
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    angle_rad = np.arctan2(nv[1], nv[0]) if len(nv) >= 2 else 0.0
    angle_deg = np.degrees(angle_rad)

    return {
        "normal_vector": nv,
        "angle_degrees": angle_deg,
        "angle_radians": angle_rad,
        "r_squared": r2,
        "coefficients": beta,
    }


def cutting_lines(
    normals: NDArray,
    cutpoints: NDArray,
    xlim: tuple = (-1.0, 1.0),
) -> dict:
    """Compute cutting lines for Coombs mesh visualization (Section 5.4).

    Each roll call defines a cutting line (hyperplane) in the policy space.
    The normal vector determines direction and the cutpoint determines offset.

    :param normals: (n_votes x n_dims) normal vectors per vote.
    :param cutpoints: (n_votes,) cutpoint offsets.
    :param xlim: x-axis limits for line endpoints.
    :return: dict with line_endpoints, angles, midpoints.
    """
    normals = np.asarray(normals, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    n_votes = normals.shape[0]

    endpoints = []
    angles = []
    midpoints = []

    for k in range(n_votes):
        nv = normals[k]
        cp = cutpoints[k]

        if abs(nv[1]) > 1e-10:
            x1, x2 = xlim
            y1 = (cp - nv[0] * x1) / nv[1]
            y2 = (cp - nv[0] * x2) / nv[1]
            endpoints.append(((x1, y1), (x2, y2)))
            midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
        elif abs(nv[0]) > 1e-10:
            x_cut = cp / nv[0]
            endpoints.append(((x_cut, -10), (x_cut, 10)))
            midpoints.append((x_cut, 0))
        else:
            endpoints.append(((0, 0), (0, 0)))
            midpoints.append((0, 0))

        angle = np.degrees(np.arctan2(nv[1], nv[0]))
        angles.append(angle)

    return {
        "endpoints": endpoints,
        "angles": np.array(angles),
        "midpoints": midpoints,
        "n_lines": n_votes,
    }


def dw_nominate(
    votes: NDArray,
    n_dims: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """DW-NOMINATE dynamic weighted estimation (Section 5.3.3).

    Dynamic Weighted NOMINATE uses normal (Gaussian) errors instead of logit,
    enabling comparable scores across legislative sessions.

    :param votes: (n_legislators x n_votes) binary vote matrix (1=Yea, 0=Nay, NaN=missing).
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: dict with ideal_points, dim_weights, normal_vectors, cutpoints, log_lik, gmp.
    """
    from morie.fn._stats_core import norm as normal_dist

    votes = np.asarray(votes, dtype=float)
    n_leg, n_votes = votes.shape
    mask = ~np.isnan(votes)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_leg, n_dims)) * 0.5
    w = np.ones(n_dims) / n_dims
    beta = 15.0

    nv = rng.standard_normal((n_votes, n_dims))
    for j in range(n_votes):
        nv[j] /= np.linalg.norm(nv[j]) + 1e-12
    mid = np.zeros((n_votes, n_dims))
    ll_prev = None

    for _iteration in range(max_iter):
        ll_old = 0.0

        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() < 2:
                continue
            X_v = X[valid]
            y_v = votes[valid, j]

            d_yea = np.sum(w * (X_v - (mid[j] + 0.5 * nv[j])) ** 2, axis=1)
            d_nay = np.sum(w * (X_v - (mid[j] - 0.5 * nv[j])) ** 2, axis=1)
            u_diff = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
            p = normal_dist.cdf(u_diff)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll_old += np.sum(y_v * np.log(p) + (1 - y_v) * np.log(1 - p))

        # converged when the log-likelihood stops moving by more than tol
        if ll_prev is not None and abs(ll_old - ll_prev) <= tol * max(1.0, abs(ll_prev)):
            break
        ll_prev = ll_old

        for j in range(n_votes):
            valid = mask[:, j]
            if valid.sum() < 2:
                continue
            X_v = X[valid]
            y_v = votes[valid, j]

            yea_center = X_v[y_v == 1].mean(axis=0) if (y_v == 1).sum() > 0 else mid[j]
            nay_center = X_v[y_v == 0].mean(axis=0) if (y_v == 0).sum() > 0 else mid[j]
            direction = yea_center - nay_center
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                nv[j] = direction / norm
            mid[j] = (yea_center + nay_center) / 2

        for i in range(n_leg):
            valid = mask[i]
            if valid.sum() < 2:
                continue
            y_i = votes[i, valid]
            nv_i = nv[valid]
            mid_i = mid[valid]

            yea_pos = mid_i + 0.5 * nv_i
            nay_pos = mid_i - 0.5 * nv_i
            target = np.where(y_i[:, None] == 1, yea_pos, nay_pos)
            X[i] = target.mean(axis=0)

        norm_x = np.linalg.norm(X, axis=1, keepdims=True)
        max_norm = norm_x.max()
        if max_norm > 1.0:
            X = X / max_norm

    ll_final = 0.0
    total = 0
    correct = 0
    for j in range(n_votes):
        valid = mask[:, j]
        if valid.sum() == 0:
            continue
        X_v = X[valid]
        y_v = votes[valid, j]
        d_yea = np.sum(w * (X_v - (mid[j] + 0.5 * nv[j])) ** 2, axis=1)
        d_nay = np.sum(w * (X_v - (mid[j] - 0.5 * nv[j])) ** 2, axis=1)
        u_diff = beta * (np.exp(-0.5 * d_yea) - np.exp(-0.5 * d_nay))
        p = normal_dist.cdf(u_diff)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll_final += np.sum(y_v * np.log(p) + (1 - y_v) * np.log(1 - p))
        pred = (p > 0.5).astype(float)
        correct += np.sum(pred == y_v)
        total += len(y_v)

    gmp = correct / total if total > 0 else 0.0
    cp = np.array([np.dot(nv[j], mid[j]) for j in range(n_votes)])

    return {
        "ideal_points": X,
        "dim_weights": w,
        "normal_vectors": nv,
        "cutpoints": cp,
        "log_lik": ll_final,
        "gmp": gmp,
        "n_dims": n_dims,
    }


def nominate_bootstrap(
    votes: NDArray,
    ideal_points: NDArray,
    normal_vectors_arr: NDArray,
    cutpoints: NDArray,
    n_boot: int = 100,
    seed: int = 42,
) -> dict:
    """Parametric bootstrap for NOMINATE standard errors (Section 5.3.1).

    Lewis and Poole (2004): generate new roll call matrices from fitted
    probabilities, re-estimate, compute SE from bootstrap distribution.

    :param votes: (n_leg x n_votes) original vote matrix.
    :param ideal_points: (n_leg x n_dims) estimated ideal points.
    :param normal_vectors_arr: (n_votes x n_dims) estimated normal vectors.
    :param cutpoints: (n_votes,) estimated cutpoints.
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: dict with se_ideal_points, boot_means, boot_samples.
    """
    votes = np.asarray(votes, dtype=float)
    X = np.asarray(ideal_points, dtype=float)
    nv = np.asarray(normal_vectors_arr, dtype=float)
    cp = np.asarray(cutpoints, dtype=float)
    n_leg, n_votes = votes.shape
    n_dims = X.shape[1]
    mask = ~np.isnan(votes)
    rng = np.random.default_rng(seed)

    beta = 15.0
    probs = np.full_like(votes, 0.5)
    for j in range(n_votes):
        proj = X @ nv[j]
        u = beta * (proj - cp[j])
        probs[:, j] = 1.0 / (1.0 + np.exp(-u))

    boot_points = np.zeros((n_boot, n_leg, n_dims))
    for b in range(n_boot):
        sim_votes = (rng.random(votes.shape) < probs).astype(float)
        sim_votes[~mask] = np.nan

        X_b = X + rng.standard_normal(X.shape) * 0.1
        for _ in range(20):
            for i in range(n_leg):
                valid = ~np.isnan(sim_votes[i])
                if valid.sum() < 2:
                    continue
                y_i = sim_votes[i, valid]
                nv_i = nv[valid]
                cp_i = cp[valid]
                proj = X_b[i] @ nv_i.T
                residual = y_i - 1.0 / (1.0 + np.exp(-beta * (proj - cp_i)))
                grad = nv_i.T @ residual
                X_b[i] += 0.01 * grad

        boot_points[b] = X_b

    se = boot_points.std(axis=0)
    boot_mean = boot_points.mean(axis=0)

    return {
        "se_ideal_points": se,
        "boot_means": boot_mean,
        "n_boot": n_boot,
    }


def alpha_nominate(
    votes: NDArray,
    n_dims: int = 1,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
    thin: int = 1,
    lop: float = 0.025,
    minvotes: int = 20,
    polarity: int = 0,
    constrain: bool = False,
) -> dict:
    """Alpha-NOMINATE (Carroll et al. 2013) by slice-within-Gibbs.

    P(yea) = Phi(Q + alpha (G - Q)), Q the quadratic and G the Gaussian
    utility difference, weight fixed at 0.5, flat priors on beta > 0 and
    alpha in [0, 1], inverse-Wishart scale penalties on the coordinates as
    in the anominate reference code.  Roll calls with minority share below
    ``lop`` and legislators with fewer than ``minvotes`` kept votes are
    dropped first; ``polarity`` (0-based) is the legislator put on the
    positive side.

    :return: dict with ideal_points, ideal_sd, yea_locations,
        nay_locations, alpha, alpha_interval, beta, draws,
        legislators_used, votes_used, n_dims, engine.
    """
    from morie._bayes_scaling import alpha_nominate as _an

    return _arrays(
        _an(
            np.asarray(votes, dtype=float).tolist(),
            n_dims=n_dims,
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            seed=seed,
            thin=int(thin),
            lop=lop,
            minvotes=int(minvotes),
            polarity=polarity,
            constrain=bool(constrain),
        ),
        ["ideal_points", "ideal_sd", "yea_locations", "nay_locations"],
    )


def ordinal_irt(
    Y: NDArray,
    n_dims: int = 1,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
    lambda_prior_precision: float = 0.0,
) -> dict:
    """Ordinal IRT: the ordinal factor model of Quinn (2004).

    y*_ij = lambda_j0 + lambda_j' phi_i + e_ij with item-specific cutpoints
    (the first fixed at 0), phi ~ N(0, I), loadings N(0, I / L0) (L0 = 0 is
    MCMCpack's flat prior), Cowles (1996) cutpoint steps.  Each item's
    categories are its observed values in order.

    :return: dict with ideal_points, ideal_sd, discrimination, intercept,
        cutpoints, acceptance, n_samples, engine.
    """
    from morie._bayes_scaling import ordinal_irt as _oi

    return _arrays(
        _oi(
            np.asarray(Y, dtype=float).tolist(),
            n_dims=int(n_dims),
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            L0=float(lambda_prior_precision),
            seed=seed,
        ),
        ["ideal_points", "ideal_sd", "discrimination", "intercept"],
    )


def dynamic_irt(
    votes: NDArray,
    time_periods: NDArray,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
    thin: int = 1,
    tau2: float = 1.0,
    c0: float = -1.0,
    d0: float = -1.0,
    e0: float = 0.0,
    E0: float = 1.0,
    a0: float = 0.0,
    A0: float = 0.1,
    b0: float = 0.0,
    B0: float = 0.1,
    anchor: int | None = None,
) -> dict:
    """Dynamic IRT of Martin & Quinn (2002), the model of MCMCdynamicIRT1d.

    z_jk = -alpha_k + beta_k theta_{j,t(k)} + e_jk (probit), ideal points
    following a random walk theta_{j,t} ~ N(theta_{j,t-1}, tau2_j) from
    theta_{j,0} ~ N(e0, E0); tau2_j ~ IG(c0/2, d0/2) when c0, d0 > 0, held
    at ``tau2`` otherwise.  Gibbs with forward-filter backward-sample paths.
    ``anchor`` (0-based) is reflected onto the positive side.

    :return: dict with theta (legislators by periods), theta_sd,
        ideal_trajectories (the same as an array), alpha, beta, tau2,
        periods, n_samples, anchor, engine.
    """
    from morie._bayes_scaling import dynamic_irt as _dyn

    res = _dyn(
        np.asarray(votes, dtype=float).tolist(),
        [int(t) for t in time_periods],
        n_samples=int(n_samples),
        burn_in=int(burn_in),
        thin=int(thin),
        seed=seed,
        tau2=tau2,
        e0=e0,
        E0=E0,
        a0=a0,
        A0=A0,
        b0=b0,
        B0=B0,
        c0=c0,
        d0=d0,
        anchor=anchor,
    )
    res["ideal_trajectories"] = np.asarray(res["theta"])
    return _arrays(res, ["theta", "theta_sd", "alpha", "beta", "tau2"])


def em_irt(
    votes,
    n_dims: int = 1,
    max_iter: int = 500,
    tol: float = 1e-6,
    *,
    x_prior=(0.0, 1.0),
    beta_prior=(0.0, 25.0),
    start=None,
    conv: str = "cor",
) -> dict:
    """EM for the binary probit IRT model (Imai, Lo and Olmsted 2016).

    ``y*_ij = alpha_j + beta_j' x_i + e_ij`` with ``e_ij ~ N(0, 1)`` and a
    yea when ``y*_ij > 0``; priors ``x_i ~ N(mu_x, s2_x I)`` and
    ``(alpha_j, beta_j) ~ N(mu_b, s2_b I)``. The E-step replaces ``y*`` by
    its truncated-normal mean (untruncated when the vote is missing); the
    M-step is the pair of ridge regressions of ``emIRT::binIRT`` with
    ``asEM = TRUE``. The fixed point is the posterior mode.

    :param votes: (n_leg x n_votes) matrix of 1 (yea), 0 (nay), NaN (missing).
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence threshold.
    :param x_prior: ``(mu_x, s2_x)``.
    :param beta_prior: ``(mu_b, s2_b)`` for the intercept and slopes.
    :param start: Optional ``(alpha, beta, x)`` starting values; default
        alpha = beta = 0 and x from 50 steps of orthogonal iteration on
        ``Y Y'`` (Y the column-centred +1/-1 votes, missing 0).
    :param conv: ``"cor"`` (1 - smallest correlation of old and new
        estimates, per column) or ``"abs"`` (largest absolute change).
    :return: dict with ideal_points, discrimination, difficulty (the
        intercepts alpha), log_lik, iterations, converged.
    """
    Y = [[float(v) for v in row] for row in votes]
    N, J, D = len(Y), len(Y[0]), int(n_dims)
    obs = [[v == v for v in row] for row in Y]
    mx, sx = float(x_prior[0]), float(x_prior[1])
    mb, sb = float(beta_prior[0]), float(beta_prior[1])
    if start is None:
        a = [0.0] * J
        b = [[0.0] * D for _ in range(J)]
        x = _em_irt_start(Y, obs, D)
    else:
        a = [float(v) for v in start[0]]
        b = [[float(v) for v in row] for row in start[1]]
        x = [[float(v) for v in row] for row in start[2]]
    it, converged = 0, False
    while it < max_iter:
        it += 1
        ys = [[_em_ystar(a[j] + _em_dot(b[j], x[i]), Y[i][j], obs[i][j]) for j in range(J)] for i in range(N)]
        x2 = [[1.0] + x[i] for i in range(N)]
        P = [[(1.0 / sb if r == c else 0.0) + _em_col(x2, r, c) for c in range(D + 1)] for r in range(D + 1)]
        B = _em_inverse(P)
        ab = []
        for j in range(J):
            rhs = [mb / sb + _em_ssum(x2[i][r] * ys[i][j] for i in range(N)) for r in range(D + 1)]
            ab.append([_em_ssum(B[r][c] * rhs[c] for c in range(D + 1)) for r in range(D + 1)])
        a_new = [ab[j][0] for j in range(J)]
        b_new = [ab[j][1:] for j in range(J)]
        eba = [_em_ssum(b_new[j][d] * a_new[j] for j in range(J)) for d in range(D)]
        Q = [[(1.0 / sx if r == c else 0.0) + _em_col(b_new, r, c) for c in range(D)] for r in range(D)]
        A = _em_inverse(Q)
        x_new = []
        for i in range(N):
            rhs = [mx / sx + _em_ssum(b_new[j][d] * ys[i][j] for j in range(J)) - eba[d] for d in range(D)]
            x_new.append([_em_ssum(A[r][c] * rhs[c] for c in range(D)) for r in range(D)])
        if it > 1:
            dev = max(
                _em_dev(x, x_new, conv),
                _em_dev([[v] for v in a], [[v] for v in a_new], conv),
                _em_dev(b, b_new, conv),
            )
            converged = dev < tol
        a, b, x = a_new, b_new, x_new
        if converged:
            break
    ll = 0.0
    for i in range(N):
        for j in range(J):
            if obs[i][j]:
                m = a[j] + _em_dot(b[j], x[i])
                ll += math.log(_em_phi_cdf(m if Y[i][j] == 1 else -m))
    return {
        "ideal_points": x,
        "discrimination": b,
        "difficulty": a,
        "log_lik": ll,
        "iterations": it,
        "converged": converged,
    }


def _em_ssum(it):
    s = 0.0
    for v in it:
        s += v
    return s


def _em_dot(u, v):
    return _em_ssum(p * q for p, q in zip(u, v))


def _em_col(M, r, c):
    return _em_ssum(row[r] * row[c] for row in M)


def _em_phi_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _em_ystar(m, y, observed):
    if not observed:
        return m
    phi = math.exp(-0.5 * m * m) / math.sqrt(2.0 * math.pi)
    if y == 1:
        return m + phi / _em_phi_cdf(m)
    return m - phi / _em_phi_cdf(-m)


def _em_inverse(M):
    n = len(M)
    W = [list(M[r]) + [1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda r: abs(W[r][k]))
        W[k], W[p] = W[p], W[k]
        piv = W[k][k]
        W[k] = [v / piv for v in W[k]]
        for r in range(n):
            if r != k and W[r][k] != 0.0:
                f = W[r][k]
                W[r] = [v - f * w for v, w in zip(W[r], W[k])]
    return [row[n:] for row in W]


def _em_dev(old, new, conv):
    if conv == "abs":
        return max(abs(u - v) for ro, rn in zip(old, new) for u, v in zip(ro, rn))
    worst = 0.0
    for d in range(len(old[0])):
        u = [r[d] for r in old]
        v = [r[d] for r in new]
        mu, mv = _em_ssum(u) / len(u), _em_ssum(v) / len(v)
        su = _em_ssum((p - mu) ** 2 for p in u)
        sv = _em_ssum((q - mv) ** 2 for q in v)
        cuv = _em_ssum((p - mu) * (q - mv) for p, q in zip(u, v))
        worst = max(worst, 1.0 - cuv / math.sqrt(su * sv) if su > 0 and sv > 0 else 1.0)
    return worst


def _em_irt_start(Y, obs, D):
    N, J = len(Y), len(Y[0])
    Z = [[(1.0 if Y[i][j] == 1 else -1.0) if obs[i][j] else 0.0 for j in range(J)] for i in range(N)]
    for j in range(J):
        c = _em_ssum(Z[i][j] for i in range(N)) / N
        for i in range(N):
            Z[i][j] -= c
    S = [[_em_dot(Z[r], Z[c]) for c in range(N)] for r in range(N)]
    V = [[float((i + 1) ** (d + 1)) for d in range(D)] for i in range(N)]
    for _ in range(50):
        V = [[_em_ssum(S[r][k] * V[k][d] for k in range(N)) for d in range(D)] for r in range(N)]
        for d in range(D):
            for e in range(d):
                p = _em_ssum(V[i][d] * V[i][e] for i in range(N))
                for i in range(N):
                    V[i][d] -= p * V[i][e]
            nrm = math.sqrt(_em_ssum(V[i][d] ** 2 for i in range(N)))
            for i in range(N):
                V[i][d] = V[i][d] / nrm if nrm > 0 else 0.0
    return [[V[i][d] * math.sqrt(N) for d in range(D)] for i in range(N)]


def nonparametric_bootstrap_scaling(
    Z: NDArray,
    scale_fn: str = "am",
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Nonparametric bootstrap for scaling methods (Sections 2.1.4, 2.2.3, 2.3.2).

    Efron and Tibshirani (1993) resampling for Aldrich-McKelvey, blackbox,
    and blackbox_transpose standard errors.

    :param Z: (n_respondents x n_stimuli) perception matrix.
    :param scale_fn: Which scaling function ("am", "blackbox", "blackbox_t").
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: dict with se_positions, boot_mean, ci_lower, ci_upper.
    """
    Z = np.asarray(Z, dtype=float)
    n_resp, n_stim = Z.shape
    rng = np.random.default_rng(seed)

    boot_positions = []
    for _ in range(n_boot):
        idx = rng.choice(n_resp, size=n_resp, replace=True)
        Z_b = Z[idx]
        if scale_fn == "am":
            result = aldrich_mckelvey(Z_b)
            boot_positions.append(result["zhat"])
        elif scale_fn == "blackbox":
            result = blackbox_scaling(Z_b)
            boot_positions.append(result["stimuli"])
        else:
            result = blackbox_scaling(Z_b.T)
            boot_positions.append(result["stimuli"])

    boot_arr = np.array(boot_positions)
    se = boot_arr.std(axis=0)
    boot_mean = boot_arr.mean(axis=0)
    ci_low = np.percentile(boot_arr, 2.5, axis=0)
    ci_high = np.percentile(boot_arr, 97.5, axis=0)

    return {
        "se_positions": se,
        "boot_mean": boot_mean,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_boot": n_boot,
    }


def wordfish_irt(
    dtm: NDArray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Wordfish / Poisson IRT for text analysis (Section 6.7, Eqs 6.48-6.49).

    Slapin and Proksch (2008): Poisson IRT model for document-feature matrices.
    Estimates document positions from word counts.

    :param dtm: (n_docs x n_words) document-term count matrix.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: dict with positions, word_weights, word_fixed, log_lik, iterations.
    """
    dtm = np.asarray(dtm, dtype=float)
    n_docs, n_words = dtm.shape

    rng = np.random.default_rng(42)
    omega = rng.standard_normal(n_docs) * 0.5
    psi = np.log(dtm.sum(axis=1) + 1)
    alpha = np.log(dtm.sum(axis=0) / dtm.sum() + 1e-10)
    beta = rng.standard_normal(n_words) * 0.1

    for iteration in range(max_iter):  # noqa: B007 -- read after the loop (iteration count)
        omega_old = omega.copy()

        for i in range(n_docs):
            eta = psi[i] + alpha + beta * omega[i]
            mu = np.exp(np.clip(eta, -20, 20))
            g = np.sum(beta * (dtm[i] - mu))
            h = -np.sum(beta**2 * mu) - 1.0
            omega[i] -= g / h

        # v0.9.5.6+: couple the omega standardisation with an alpha +
        # beta rescale so eta = psi + alpha + beta*omega is invariant.
        # Pre-v0.9.5.6 only standardised omega, silently distorting
        # beta and degrading convergence.
        m_om = omega.mean()
        s_om = omega.std() + 1e-12
        omega = (omega - m_om) / s_om
        alpha = alpha + beta * m_om  # absorb mean shift
        beta = beta * s_om  # rescale to standardised omega

        for j in range(n_words):
            eta = psi + alpha[j] + beta[j] * omega
            mu = np.exp(np.clip(eta, -20, 20))
            g_a = np.sum(dtm[:, j] - mu)
            h_a = -np.sum(mu) - 0.01
            alpha[j] -= g_a / h_a

            g_b = np.sum(omega * (dtm[:, j] - mu))
            h_b = -np.sum(omega**2 * mu) - 0.01
            beta[j] -= g_b / h_b

        change = np.linalg.norm(omega - omega_old) / (np.linalg.norm(omega_old) + 1e-12)
        if change < tol:
            break

    ll = 0.0
    for i in range(n_docs):
        eta = psi[i] + alpha + beta * omega[i]
        mu = np.exp(np.clip(eta, -20, 20))
        ll += np.sum(dtm[i] * np.log(mu + 1e-15) - mu)

    return {
        "positions": omega,
        "word_weights": beta,
        "word_fixed": alpha,
        "doc_fixed": psi,
        "log_lik": ll,
        "iterations": iteration + 1,
    }
