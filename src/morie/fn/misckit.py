# morie.fn -- function file (rootcoder007/morie)
"""Assorted methods: the police-reported Crime Severity Index and its offence weights, the
U-learner for heterogeneous treatment effects, repeated k-fold index sets, FastICA (parallel,
log-cosh) and rStress multidimensional scaling."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .lbfgsb import lbfgsb_minimize

__all__ = ["csi_weights", "crime_severity_index", "u_learner_cate", "repeated_kfold_indices", "fast_ica", "rstress_mds"]


def csi_weights(incarceration_rate, mean_sentence_days) -> list:
    r"""Offence seriousness weights of the Crime Severity Index.

    ``w_i = (incarceration rate of offence i) x (mean prison sentence in
    days of offence i)`` (Babyak et al. 2009, section 2.3).

    References
    ----------
    Babyak, C., Alavi, A., Collins, K., Halladay, A. and Tapper, D. (2009).
    The methodology of the police-reported Crime Severity Index. SSC Annual
    Meeting, Proceedings of the Survey Methods Section.

    Examples
    --------
    >>> csi_weights([0.5, 0.1], [300.0, 20.0])
    [150.0, 2.0]
    """
    return [float(a) * float(b) for a, b in zip(incarceration_rate, mean_sentence_days)]


def crime_severity_index(counts, weights, population, *, base: int = 0, per: float = 100000.0) -> RichResult:
    r"""Police-reported Crime Severity Index: a weighted volume index normalised to a base period.

    ``counts[t][i]`` offences of type ``i`` in period ``t``; the weighted
    rate is ``R_t = sum_i w_i n_ti / P_t x per``, and the index
    ``CSI_t = 100 R_t / R_base`` (Babyak et al. 2009, eq. 2.1).

    Examples
    --------
    >>> r = crime_severity_index([[10, 2], [8, 3]], [1.0, 50.0], [1000.0, 1000.0])
    >>> r.index
    [100.0, 143.63636363636363]
    """
    w = [float(v) for v in weights]
    R = [ssum(w[i] * float(row[i]) for i in range(len(w))) / float(p) * per for row, p in zip(counts, population)]
    return RichResult(payload={"index": [100.0 * v / R[base] for v in R], "weighted_rate": R})


def _ridge(X, y, lam):
    p = len(X[0])
    G = [[ssum(r[a] * r[b] for r in X) + (lam if (a == b and a > 0) else 0.0) for b in range(p)] for a in range(p)]
    return solve(G, [ssum(r[a] * v for r, v in zip(X, y)) for a in range(p)])


def _logit_ridge(X, d, lam, iters=50):
    p = len(X[0])
    b = [0.0] * p
    for _ in range(iters):
        eta = [ssum(r[k] * b[k] for k in range(p)) for r in X]
        pr = [1.0 / (1.0 + math.exp(-e)) for e in eta]
        g = [ssum(r[a] * (dv - q) for r, dv, q in zip(X, d, pr)) - (lam * b[a] if a > 0 else 0.0) for a in range(p)]
        H = [
            [
                ssum(r[a] * r[c] * q * (1 - q) for r, q in zip(X, pr)) + (lam if (a == c and a > 0) else 0.0)
                for c in range(p)
            ]
            for a in range(p)
        ]
        step = solve(H, g)
        b = [u + v for u, v in zip(b, step)]
        if max(abs(v) for v in step) <= 1e-12:
            break
    return b


def u_learner_cate(y, d, X, *, folds: int = 5, l2: float = 1.0, clip: float = 0.01, seed: int = 0) -> RichResult:
    r"""U-learner of the conditional average treatment effect (Kunzel et al. 2019).

    With cross-fitted outcome ``m(x) = E(Y | X)`` (ridge regression) and
    propensity ``e(x) = P(D = 1 | X)`` (ridge logistic regression), the
    pseudo-outcome ``U = (Y - m(X)) / (D - e(X))`` (denominator clipped away
    from 0 by ``clip``) is regressed on ``X`` by ridge regression, whose fit is
    ``tau(x)``. Folds are a Philox permutation dealt in turn.

    References
    ----------
    Kunzel, S. R., Sekhon, J. S., Bickel, P. J. and Yu, B. (2019).
    Metalearners for estimating heterogeneous treatment effects using machine
    learning. PNAS 116, 4156-4165. Nie, X. and Wager, S. (2021).
    Quasi-oracle estimation of heterogeneous treatment effects. Biometrika 108, 299-319.

    Examples
    --------
    >>> X = [[math.sin(i)] for i in range(40)]
    >>> d = [i % 2 for i in range(40)]
    >>> y = [2.0 * dv + x[0] for dv, x in zip(d, X)]
    >>> r = u_learner_cate(y, d, X, l2=1e-8)
    >>> round(r.coefficients[0], 4)
    2.0
    """
    n = len(y)
    Xr = [[1.0] + [float(v) for v in r] for r in X]
    ys = [float(v) for v in y]
    ds = [float(v) for v in d]
    perm = list(range(n))
    u = random_uniform(n, seed=seed, stream=0)
    for i in range(n - 1, 0, -1):
        j = int(math.floor(float(u[i]) * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    fold = [0] * n
    for pos, i in enumerate(perm):
        fold[i] = pos % folds
    mhat, ehat = [0.0] * n, [0.0] * n
    for f in range(folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        bm = _ridge([Xr[i] for i in tr], [ys[i] for i in tr], l2)
        be = _logit_ridge([Xr[i] for i in tr], [ds[i] for i in tr], l2)
        for i in te:
            mhat[i] = ssum(Xr[i][k] * bm[k] for k in range(len(bm)))
            ehat[i] = 1.0 / (1.0 + math.exp(-ssum(Xr[i][k] * be[k] for k in range(len(be)))))
    U = []
    for i in range(n):
        den = ds[i] - ehat[i]
        if abs(den) < clip:
            den = clip if den >= 0 else -clip
        U.append((ys[i] - mhat[i]) / den)
    bt = _ridge(Xr, U, l2)
    tau = [ssum(r[k] * bt[k] for k in range(len(bt))) for r in Xr]
    return RichResult(payload={"coefficients": bt, "tau": tau, "pseudo_outcome": U, "m_hat": mhat, "e_hat": ehat})


def repeated_kfold_indices(n: int, k: int = 10, repeats: int = 3, *, seed: int = 0) -> list:
    r"""Fold labels for repeated k-fold cross-validation (``repeats`` Philox permutations).

    Repeat ``r`` shuffles ``0..n-1`` by Fisher-Yates on stream ``r`` and
    deals the permuted units to folds ``0..k-1`` in turn (the resampling of
    ``caret::trainControl(method = "repeatedcv")``, unstratified).

    References
    ----------
    Kuhn, M. and Johnson, K. (2013). Applied Predictive Modeling, section 4.4.
    Brownlee, J. (2016). Machine Learning Mastery with R, ch. 15.

    Examples
    --------
    >>> f = repeated_kfold_indices(6, 3, 2)
    >>> [sorted(r).count(0) for r in f]
    [2, 2]
    """
    out = []
    for r in range(repeats):
        perm = list(range(n))
        u = random_uniform(n, seed=seed, stream=r)
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[i]) * (i + 1)))
            perm[i], perm[j] = perm[j], perm[i]
        fold = [0] * n
        for pos, i in enumerate(perm):
            fold[i] = pos % k
        out.append(fold)
    return out


def _eig_sym(M):
    w, V = np.linalg.eigh(np.asarray(M, dtype=float))
    n = len(M)
    w = [float(v) for v in w]
    order = sorted(range(n), key=lambda i: -w[i])
    vecs = []
    for i in order:
        v = [float(V[r][i]) for r in range(n)]
        big = max(range(n), key=lambda r: abs(v[r]))
        if v[big] < 0:
            v = [-a for a in v]
        vecs.append(v)
    return [w[i] for i in order], vecs


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _sym_decorr(W):
    k = len(W)
    WW = _mm(W, [list(c) for c in zip(*W)])
    w, vecs = _eig_sym(WW)
    M = [[ssum(vecs[q][a] * vecs[q][b] / math.sqrt(w[q]) for q in range(k)) for b in range(k)] for a in range(k)]
    return _mm(M, W)


def fast_ica(X, n_comp: int, *, w_init=None, alpha: float = 1.0, max_iter: int = 200, tol: float = 1e-4) -> RichResult:
    r"""FastICA, parallel (symmetric) algorithm with the log-cosh contrast (Hyvarinen 1999).

    Columns are centred and whitened by ``K = D^(-1/2) E'`` from the
    covariance eigen-decomposition (eigenvectors signed with their largest
    entry positive); from ``w_init`` (identity by default) the update
    ``W <- E(g(WZ) Z') - diag(E g'(WZ)) W`` with ``g = tanh(alpha u)`` is
    followed by symmetric decorrelation ``(W W')^(-1/2) W`` until
    ``max ||diag(W_new W')| - 1| < tol`` (the conventions of the fastICA package).
    Returns the sources ``S`` (rows are observations), ``W``, ``K`` and the
    mixing matrix ``A``.

    References
    ----------
    Hyvarinen, A. (1999). Fast and robust fixed-point algorithms for
    independent component analysis. IEEE Trans. Neural Networks 10, 626-634.
    Hyvarinen, A. and Oja, E. (2000). Independent component analysis:
    algorithms and applications. Neural Networks 13, 411-430.

    Examples
    --------
    >>> X = [[math.sin(i * 0.3) + 0.5 * ((i % 7) - 3), math.sin(i * 0.3) - 0.5 * ((i % 7) - 3)] for i in range(60)]
    >>> r = fast_ica(X, 2)
    >>> len(r.S), len(r.S[0])
    (60, 2)
    """
    n, p = len(X), len(X[0])
    mu = [ssum(r[j] for r in X) / n for j in range(p)]
    Xc = [[r[j] - mu[j] for j in range(p)] for r in X]
    V = [[ssum(r[a] * r[b] for r in Xc) / n for b in range(p)] for a in range(p)]
    w, vecs = _eig_sym(V)
    K = [[vecs[q][j] / math.sqrt(w[q]) for j in range(p)] for q in range(n_comp)]
    Z = _mm(K, [list(c) for c in zip(*Xc)])  # n_comp x n
    W = (
        [[float(v) for v in r] for r in w_init]
        if w_init is not None
        else [[1.0 if a == b else 0.0 for b in range(n_comp)] for a in range(n_comp)]
    )
    W = _sym_decorr(W)
    it = 0
    for _ in range(max_iter):
        it += 1
        WZ = _mm(W, Z)
        G = [[math.tanh(alpha * v) for v in r] for r in WZ]
        v1 = [[ssum(G[a][t] * Z[b][t] for t in range(n)) / n for b in range(n_comp)] for a in range(n_comp)]
        gp = [ssum(alpha * (1 - G[a][t] ** 2) for t in range(n)) / n for a in range(n_comp)]
        W1 = [[v1[a][b] - gp[a] * W[a][b] for b in range(n_comp)] for a in range(n_comp)]
        W1 = _sym_decorr(W1)
        lim = max(abs(abs(ssum(W1[a][q] * W[a][q] for q in range(n_comp))) - 1) for a in range(n_comp))
        W = W1
        if lim < tol:
            break
    WK = _mm(W, K)
    S = [list(c) for c in zip(*_mm(WK, [list(c) for c in zip(*Xc)]))]
    A = inverse(WK) if n_comp == p else None
    return RichResult(payload={"S": S, "W": W, "K": K, "A": A, "iterations": it})


def rstress_mds(delta, *, r: float = 0.5, ndim: int = 2, weights=None, max_iter: int = 2000) -> RichResult:
    r"""rStress multidimensional scaling: minimise ``sum_{i<j} w_ij (delta_ij - d_ij(X)^r)^2``.

    Classical (Torgerson) scaling of ``delta^(1/r)`` starts L-BFGS on the
    configuration with the analytic gradient; ``r = 1`` is Kruskal's raw
    stress, ``r = 1/2`` fits squared-distance powers (de Leeuw, Groenen and
    Mair 2016). Reports raw and normalised stress
    ``sqrt(raw / sum w delta^2)``; the configuration is centred.

    References
    ----------
    de Leeuw, J., Groenen, P. J. F. and Mair, P. (2016). Minimizing rStress
    using majorization. Technical report, UCLA. Kruskal, J. B. (1964).
    Psychometrika 29, 1-27.

    Examples
    --------
    >>> D = [[0, 1, 1, 2 ** 0.5], [1, 0, 2 ** 0.5, 1], [1, 2 ** 0.5, 0, 1], [2 ** 0.5, 1, 1, 0]]
    >>> r = rstress_mds(D, r=1.0)
    >>> round(r.stress, 8)
    0.0
    """
    Dl = [[float(v) for v in row] for row in delta]
    n = len(Dl)
    Wt = (
        [[1.0 if i != j else 0.0 for j in range(n)] for i in range(n)]
        if weights is None
        else [[float(v) for v in row] for row in weights]
    )
    B0 = [[-0.5 * Dl[i][j] ** (2.0 / r) for j in range(n)] for i in range(n)]
    rm = [ssum(row) / n for row in B0]
    gm = ssum(rm) / n
    B = [[B0[i][j] - rm[i] - rm[j] + gm for j in range(n)] for i in range(n)]
    w, vecs = _eig_sym(B)
    x0 = [vecs[k][i] * math.sqrt(max(w[k], 0.0)) for i in range(n) for k in range(ndim)]

    def fg(x):
        f = 0.0
        g = [0.0] * len(x)
        for i in range(n):
            for j in range(i + 1, n):
                if Wt[i][j] == 0:
                    continue
                diff = [x[i * ndim + k] - x[j * ndim + k] for k in range(ndim)]
                d = math.sqrt(ssum(v * v for v in diff))
                dr = d**r
                res = Dl[i][j] - dr
                f += Wt[i][j] * res * res
                if d > 0:
                    c = -2 * Wt[i][j] * res * r * d ** (r - 2)
                    for k in range(ndim):
                        g[i * ndim + k] += c * diff[k]
                        g[j * ndim + k] -= c * diff[k]
        return f, g

    res = lbfgsb_minimize(
        lambda v: fg(list(v))[0], x0, grad=lambda v: fg(list(v))[1], pgtol=1e-10, factr=10.0, max_iter=max_iter
    )
    x = [float(v) for v in res.x]
    raw = fg(x)[0]
    cm = [ssum(x[i * ndim + k] for i in range(n)) / n for k in range(ndim)]
    conf = [[x[i * ndim + k] - cm[k] for k in range(ndim)] for i in range(n)]
    norm = ssum(Wt[i][j] * Dl[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
    return RichResult(payload={"conf": conf, "raw_stress": raw, "stress": math.sqrt(raw / norm), "r": r})


def cheatsheet() -> str:
    return (
        "csi_weights / crime_severity_index / u_learner_cate / repeated_kfold_indices / fast_ica / rstress_mds -> "
        "crime severity, meta-learners, resampling, ICA and rStress MDS."
    )
