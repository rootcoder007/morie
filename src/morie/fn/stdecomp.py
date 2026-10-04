# morie.fn -- function file (rootcoder007/morie)
"""Space-time decompositions: CP (PARAFAC) and Tucker (HOSVD/HOOI) tensor decompositions,
regularised matrix factorisation of space-time fields with missing values, Haar wavelet
transforms and multiresolution analyses in one and two dimensions, wavelet detrending and
harmonic (Fourier) trend regression."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult

__all__ = [
    "cp_als",
    "tucker_hooi",
    "matrix_factorization_als",
    "haar_dwt",
    "haar_mra",
    "haar_dwt_2d",
    "haar_mra_2d",
    "wavelet_detrend",
    "harmonic_regression",
]


def _eig_desc(M):
    w, V = np.linalg.eigh(np.asarray(M, dtype=float))
    w = [float(v) for v in w]
    n = len(w)
    order = sorted(range(n), key=lambda i: -w[i])
    vecs = []
    for i in order:
        v = [float(V[r][i]) for r in range(n)]
        k = max(range(n), key=lambda r: abs(v[r]))
        if v[k] < 0:
            v = [-a for a in v]
        vecs.append(v)
    return [w[i] for i in order], vecs


def _unfold(T, mode):
    n1, J, K = len(T), len(T[0]), len(T[0][0])
    if mode == 0:
        return [[T[i][j][k] for k in range(K) for j in range(J)] for i in range(n1)]
    if mode == 1:
        return [[T[i][j][k] for k in range(K) for i in range(n1)] for j in range(J)]
    return [[T[i][j][k] for j in range(J) for i in range(n1)] for k in range(K)]


def _gram(M):
    return [[ssum(a * b for a, b in zip(r, s)) for s in M] for r in M]


def _lead(M, r):
    return _eig_desc(_gram(M))[1][:r]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def cp_als(tensor, rank: int, *, max_iter: int = 500, tol: float = 1e-12) -> RichResult:
    r"""CP (CANDECOMP/PARAFAC) decomposition of a three-way array by alternating least squares.

    ``X[i][j][k] ~ sum_r lambda_r A[i][r] B[j][r] C[k][r]``. Factors start
    from the leading eigenvectors of the unfolding Gram matrices (HOSVD
    start); each update solves e.g.
    ``A = X_(1) (C kr B) ((C'C) * (B'B))^(-1)`` (Khatri-Rao ``kr``, Hadamard
    ``*``), until the relative change of the fit is below ``tol``. Columns
    are normalised to unit length (weights ``lambda``), ordered by weight,
    with the largest entry of each ``A`` and ``B`` column positive.

    References
    ----------
    Kolda, T. G. and Bader, B. W. (2009). Tensor decompositions and
    applications. SIAM Review 51, 455-500. Harshman, R. A. (1970).
    Foundations of the PARAFAC procedure. UCLA Working Papers in Phonetics 16.

    Examples
    --------
    >>> T = [[[a * b * c for c in (1.0, 2.0)] for b in (1.0, 3.0)] for a in (1.0, 2.0, 2.0)]
    >>> r = cp_als(T, 1)
    >>> round(r.weights[0], 10), round(r.fit, 12)
    (21.2132034356, 1.0)
    """
    T = [[[float(v) for v in row] for row in sl] for sl in tensor]
    n1, J, K = len(T), len(T[0]), len(T[0][0])
    X0, X1, X2 = _unfold(T, 0), _unfold(T, 1), _unfold(T, 2)
    Ac, Bc, Cc = _lead(X0, rank), _lead(X1, rank), _lead(X2, rank)
    A = [[Ac[r][i] if r < len(Ac) else 0.0 for r in range(rank)] for i in range(n1)]
    B = [[Bc[r][j] if r < len(Bc) else 0.0 for r in range(rank)] for j in range(J)]
    C = [[Cc[r][k] if r < len(Cc) else 0.0 for r in range(rank)] for k in range(K)]
    norm2 = ssum(v * v for sl in T for row in sl for v in row)

    def kr(U, V):
        # Khatri-Rao product, rows indexed (u, v) with v fastest... matched to the unfoldings
        return [[U[a][r] * V[b][r] for r in range(rank)] for a in range(len(U)) for b in range(len(V))]

    def update(Xn, P, Q):
        # P outer (slow) index, Q inner (fast) index of the unfolding columns
        G = [
            [
                ssum(P[a][r] * P[a][s] for a in range(len(P))) * ssum(Q[b][r] * Q[b][s] for b in range(len(Q)))
                for s in range(rank)
            ]
            for r in range(rank)
        ]
        M = _mm(Xn, kr(P, Q))
        Gi = inverse(G)
        return _mm(M, Gi)

    fit_old = -1.0
    fit = 0.0
    it = 0
    for _ in range(max_iter):
        it += 1
        A = update(X0, C, B)
        B = update(X1, C, A)
        C = update(X2, B, A)
        # residual norm via ||X||^2 - 2 <X, Xhat> + ||Xhat||^2
        inner = ssum(
            A[i][r] * ssum(B[j][r] * ssum(T[i][j][k] * C[k][r] for k in range(K)) for j in range(J))
            for i in range(n1)
            for r in range(rank)
        )
        G = [
            [
                ssum(A[a][r] * A[a][s] for a in range(n1))
                * ssum(B[b][r] * B[b][s] for b in range(J))
                * ssum(C[c][r] * C[c][s] for c in range(K))
                for s in range(rank)
            ]
            for r in range(rank)
        ]
        nh = ssum(v for row in G for v in row)
        res = max(norm2 - 2 * inner + nh, 0.0)
        fit = 1.0 - math.sqrt(res) / math.sqrt(norm2)
        if abs(fit - fit_old) < tol:
            break
        fit_old = fit
    lam = []
    for r in range(rank):
        na = math.sqrt(ssum(A[i][r] ** 2 for i in range(n1)))
        nb = math.sqrt(ssum(B[j][r] ** 2 for j in range(J)))
        nc = math.sqrt(ssum(C[k][r] ** 2 for k in range(K)))
        sa = 1.0 if A[max(range(n1), key=lambda i: abs(A[i][r]))][r] >= 0 else -1.0
        sb = 1.0 if B[max(range(J), key=lambda j: abs(B[j][r]))][r] >= 0 else -1.0
        for i in range(n1):
            A[i][r] = sa * A[i][r] / na
        for j in range(J):
            B[j][r] = sb * B[j][r] / nb
        for k in range(K):
            C[k][r] = sa * sb * C[k][r] / nc
        lam.append(na * nb * nc)
    order = sorted(range(rank), key=lambda r: -lam[r])
    return RichResult(
        payload={
            "weights": [lam[r] for r in order],
            "A": [[row[r] for r in order] for row in A],
            "B": [[row[r] for r in order] for row in B],
            "C": [[row[r] for r in order] for row in C],
            "fit": fit,
            "iterations": it,
        }
    )


def _mode_product_t(T, U, mode):
    # T x_mode U'  (U is n_mode x r as list of columns given as rows of Ut)
    n1, J, K = len(T), len(T[0]), len(T[0][0])
    r = len(U)
    if mode == 0:
        return [[[ssum(U[a][i] * T[i][j][k] for i in range(n1)) for k in range(K)] for j in range(J)] for a in range(r)]
    if mode == 1:
        return [[[ssum(U[b][j] * T[i][j][k] for j in range(J)) for k in range(K)] for b in range(r)] for i in range(n1)]
    return [[[ssum(U[c][k] * T[i][j][k] for k in range(K)) for c in range(r)] for j in range(J)] for i in range(n1)]


def tucker_hooi(tensor, ranks, *, max_iter: int = 50, tol: float = 1e-12) -> RichResult:
    r"""Tucker decomposition by HOSVD followed by higher-order orthogonal iteration (HOOI).

    ``X ~ G x_1 U_1 x_2 U_2 x_3 U_3`` with orthonormal factors: the HOSVD
    start takes the leading ``r_n`` eigenvectors of ``X_(n) X_(n)'``; HOOI
    replaces ``U_n`` by the leading eigenvectors of ``Y_(n) Y_(n)'`` with
    ``Y = X`` projected on the other two factors, until the core norm
    stabilises. Eigenvector signs make the largest entry positive.

    References
    ----------
    De Lathauwer, L., De Moor, B. and Vandewalle, J. (2000). A multilinear
    singular value decomposition; On the best rank-(R1, R2, ..., RN)
    approximation of higher-order tensors. SIAM J. Matrix Anal. Appl. 21,
    1253-1278 and 1324-1342. Tucker, L. R. (1966). Psychometrika 31, 279-311.

    Examples
    --------
    >>> T = [[[a * b * c for c in (1.0, 2.0)] for b in (1.0, 3.0)] for a in (1.0, 2.0, 2.0)]
    >>> r = tucker_hooi(T, (1, 1, 1))
    >>> round(abs(r.core[0][0][0]), 10), round(r.fit, 6)
    (21.2132034356, 1.0)
    """
    T = [[[float(v) for v in row] for row in sl] for sl in tensor]
    r1, r2, r3 = ranks
    U = [_lead(_unfold(T, 0), r1), _lead(_unfold(T, 1), r2), _lead(_unfold(T, 2), r3)]
    norm2 = ssum(v * v for sl in T for row in sl for v in row)
    old = -1.0
    it = 0
    for _ in range(max_iter):
        it += 1
        Y = _mode_product_t(_mode_product_t(T, U[1], 1), U[2], 2)
        U[0] = _lead(_unfold(Y, 0), r1)
        Y = _mode_product_t(_mode_product_t(T, U[0], 0), U[2], 2)
        U[1] = _lead(_unfold(Y, 1), r2)
        Y = _mode_product_t(_mode_product_t(T, U[0], 0), U[1], 1)
        U[2] = _lead(_unfold(Y, 2), r3)
        G = _mode_product_t(Y, U[2], 2)
        gn = ssum(v * v for sl in G for row in sl for v in row)
        if abs(gn - old) <= tol * norm2:
            break
        old = gn
    G = _mode_product_t(_mode_product_t(_mode_product_t(T, U[0], 0), U[1], 1), U[2], 2)
    gn = ssum(v * v for sl in G for row in sl for v in row)
    fit = 1.0 - math.sqrt(max(norm2 - gn, 0.0)) / math.sqrt(norm2)
    return RichResult(payload={"core": G, "factors": [_t(u) for u in U], "fit": fit, "iterations": it})


def matrix_factorization_als(X, rank: int, *, l2: float = 0.1, max_iter: int = 200, tol: float = 1e-12) -> RichResult:
    r"""Regularised low-rank factorisation ``X ~ U V'`` of a space-time field with missing values.

    Minimises ``sum_{observed} (x_ij - u_i'v_j)^2 + l2 (||U||^2 + ||V||^2)``
    by alternating ridge regressions (Koren, Bell and Volinsky 2009);
    missing entries are ``None``/NaN. ``V`` starts from the leading
    eigenvectors of ``X0'X0`` (``X0`` zero-filled) scaled by the square roots
    of their eigenvalues. ``fitted`` fills the gaps.

    References
    ----------
    Koren, Y., Bell, R. and Volinsky, C. (2009). Matrix factorization
    techniques for recommender systems. Computer 42(8), 30-37. Hastie, T. et
    al. (2015). Matrix completion and low-rank SVD via fast alternating least
    squares. JMLR 16, 3367-3402.

    Examples
    --------
    >>> r = matrix_factorization_als([[1.0, 2.0], [2.0, 4.0], [3.0, None]], 1, l2=0.0)
    >>> round(r.fitted[2][1], 4)
    6.0
    """
    m, n = len(X), len(X[0])
    obs = [[(v is not None and v == v) for v in row] for row in X]
    X0 = [[float(X[i][j]) if obs[i][j] else 0.0 for j in range(n)] for i in range(m)]
    w, vecs = _eig_desc(_gram(_t(X0)))
    V = [[vecs[r][j] * math.sqrt(max(w[r], 0.0)) if r < len(vecs) else 0.0 for r in range(rank)] for j in range(n)]
    U = [[0.0] * rank for _ in range(m)]

    def ridge(F, rows, y):
        G = [
            [ssum(F[a][r] * F[a][s] for a in rows) + (l2 if r == s else 0.0) for s in range(rank)] for r in range(rank)
        ]
        h = [ssum(F[a][r] * y[a] for a in rows) for r in range(rank)]
        return solve(G, h)

    old = math.inf
    it = 0
    for _ in range(max_iter):
        it += 1
        for i in range(m):
            rows = [j for j in range(n) if obs[i][j]]
            U[i] = ridge(V, rows, X0[i]) if rows else [0.0] * rank
        for j in range(n):
            rows = [i for i in range(m) if obs[i][j]]
            V[j] = ridge(U, rows, [X0[i][j] for i in range(m)]) if rows else [0.0] * rank
        loss = ssum(
            (X0[i][j] - ssum(U[i][r] * V[j][r] for r in range(rank))) ** 2
            for i in range(m)
            for j in range(n)
            if obs[i][j]
        )
        loss += l2 * (ssum(v * v for row in U for v in row) + ssum(v * v for row in V for v in row))
        if abs(old - loss) <= tol * max(1.0, loss):
            break
        old = loss
    fitted = [[ssum(U[i][r] * V[j][r] for r in range(rank)) for j in range(n)] for i in range(m)]
    return RichResult(payload={"U": U, "V": V, "fitted": fitted, "loss": loss, "iterations": it})


def _haar_step(x):
    h = len(x) // 2
    s = [(x[2 * t + 1] + x[2 * t]) / math.sqrt(2.0) for t in range(h)]
    d = [(x[2 * t + 1] - x[2 * t]) / math.sqrt(2.0) for t in range(h)]
    return s, d


def _haar_inv(s, d):
    out = []
    for a, b in zip(s, d):
        out += [(a - b) / math.sqrt(2.0), (a + b) / math.sqrt(2.0)]
    return out


def haar_dwt(x, levels: int) -> RichResult:
    r"""Orthonormal Haar discrete wavelet transform (pyramid algorithm).

    At each level ``W_t = (V_{2t+1} - V_{2t}) / sqrt(2)`` and
    ``V_t = (V_{2t+1} + V_{2t}) / sqrt(2)`` (Percival and Walden 2000
    convention, as ``wavelets::dwt(filter = "haar")``); the length must be a
    multiple of ``2^levels``.

    References
    ----------
    Percival, D. B. and Walden, A. T. (2000). Wavelet Methods for Time
    Series Analysis, ch. 4. Mallat, S. G. (1989). IEEE Trans. PAMI 11, 674-693.

    Examples
    --------
    >>> r = haar_dwt([1.0, 4.0, 2.0, 8.0], 1)
    >>> [round(v, 12) for v in r.details[0]]
    [2.12132034356, 4.242640687119]
    """
    v = [float(a) for a in x]
    W, V = [], []
    for _ in range(levels):
        v, d = _haar_step(v)
        W.append(d)
        V.append(v)
    return RichResult(payload={"details": W, "smooths": V})


def haar_mra(x, levels: int) -> RichResult:
    r"""Haar multiresolution analysis ``x = sum_j D_j + S_J`` (additive details and smooth).

    ``D_j`` is the inverse transform of the level-``j`` wavelet coefficients
    alone and ``S_J`` that of the level-``J`` scaling coefficients (as
    ``wavelets::mra(method = "dwt")``).

    Examples
    --------
    >>> r = haar_mra([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 6.0], 2)
    >>> [round(v, 12) for v in r.smooth]
    [3.75, 3.75, 3.75, 3.75, 5.25, 5.25, 5.25, 5.25]
    """
    dw = haar_dwt(x, levels)
    n = len(x)

    def rebuild(level, coefs, is_detail):
        s = [0.0] * len(coefs) if is_detail else list(coefs)
        d = list(coefs) if is_detail else [0.0] * len(coefs)
        out = _haar_inv(s, d)
        for _ in range(level - 1):
            out = _haar_inv(out, [0.0] * len(out))
        return out

    D = [rebuild(j + 1, dw.details[j], True) for j in range(levels)]
    S = rebuild(levels, dw.smooths[-1], False)
    assert len(S) == n
    return RichResult(payload={"details": D, "smooth": S})


def _rows_step(M):
    out_s, out_d = [], []
    for row in M:
        s, d = _haar_step(row)
        out_s.append(s)
        out_d.append(d)
    return out_s, out_d


def _cols_step(M):
    s, d = _rows_step(_t(M))
    return _t(s), _t(d)


def haar_dwt_2d(field, levels: int) -> RichResult:
    r"""Two-dimensional separable Haar wavelet transform of a field (space x space or space x time).

    Each level applies the Haar step along the rows and then along the
    columns of the current approximation: ``a`` = row-smooth/column-smooth,
    ``h`` = row-smooth/column-detail, ``v`` = row-detail/column-smooth and
    ``d`` = row-detail/column-detail. Energy is preserved.

    References
    ----------
    Mallat, S. G. (1989). A theory for multiresolution signal decomposition:
    the wavelet representation. IEEE Trans. PAMI 11, 674-693.

    Examples
    --------
    >>> r = haar_dwt_2d([[1.0, 2.0], [3.0, 4.0]], 1)
    >>> r.approximation, r.levels[0]["d"]
    ([[5.0]], [[0.0]])
    """
    a = [[float(v) for v in row] for row in field]
    out = []
    for _ in range(levels):
        rs, rd = _rows_step(a)
        a, h = _cols_step(rs)
        v, d = _cols_step(rd)
        out.append({"h": h, "v": v, "d": d})
    return RichResult(payload={"approximation": a, "levels": out})


def haar_mra_2d(field, levels: int) -> RichResult:
    r"""Two-dimensional Haar multiresolution analysis: ``field = sum_j (H_j + V_j + D_j) + S_J``.

    Each component is the inverse transform of one subband of
    :func:`haar_dwt_2d` with all other coefficients set to zero.

    Examples
    --------
    >>> r = haar_mra_2d([[1.0, 2.0], [3.0, 4.0]], 1)
    >>> [[round(v, 12) for v in row] for row in r.smooth]
    [[2.5, 2.5], [2.5, 2.5]]
    """
    dw = haar_dwt_2d(field, levels)

    def inv2(a, h, v, d):
        rs = _t([_haar_inv(s, dd) for s, dd in zip(_t(a), _t(h))])
        rd = _t([_haar_inv(s, dd) for s, dd in zip(_t(v), _t(d))])
        return [_haar_inv(s, dd) for s, dd in zip(rs, rd)]

    def zeros(M):
        return [[0.0] * len(M[0]) for _ in M]

    def up(M, level):
        for _ in range(level):
            M = inv2(M, zeros(M), zeros(M), zeros(M))
        return M

    comps = []
    for j, lv in enumerate(dw.levels):
        z = zeros(lv["h"])
        parts = {}
        for key in ("h", "v", "d"):
            args = {"h": z, "v": z, "d": z}
            args[key] = lv[key]
            parts[key] = up(inv2(z, args["h"], args["v"], args["d"]), j)
        comps.append(parts)
    smooth = up(dw.approximation, levels)
    return RichResult(payload={"details": comps, "smooth": smooth})


def wavelet_detrend(x, levels: int) -> RichResult:
    r"""Remove the level-``J`` Haar MRA smooth (the low-frequency trend) from a series.

    ``detrended = x - S_J`` with ``S_J`` from :func:`haar_mra`.

    Examples
    --------
    >>> [round(v, 12) for v in wavelet_detrend([1.0, 3.0, 2.0, 6.0], 1).detrended]
    [-1.0, 1.0, -2.0, 2.0]
    """
    S = haar_mra(x, levels).smooth
    return RichResult(payload={"trend": S, "detrended": [float(a) - b for a, b in zip(x, S)]})


def harmonic_regression(y, t, period: float, n_harmonics: int = 2, *, trend: bool = True) -> RichResult:
    r"""Harmonic (Fourier) trend regression by least squares.

    ``y_t = a + b t + sum_{k=1}^{K} (c_k cos(2 pi k t / P) + d_k sin(2 pi k t / P)) + e_t``
    (``b`` omitted without ``trend``). Returns coefficients, amplitudes
    ``sqrt(c_k^2 + d_k^2)``, phases ``atan2(d_k, c_k)``, fitted values and
    ``R^2``.

    References
    ----------
    Bloomfield, P. (2000). Fourier Analysis of Time Series, 2nd ed., ch. 2.
    Jakubauskas, M. E., Legates, D. R. and Kastens, J. H. (2001). Harmonic
    analysis of time-series AVHRR NDVI data. Photogramm. Eng. Remote Sens. 67, 461-470.

    Examples
    --------
    >>> t = [float(i) for i in range(12)]
    >>> y = [2 + 3 * math.cos(2 * math.pi * v / 12) for v in t]
    >>> r = harmonic_regression(y, t, 12.0, 1, trend=False)
    >>> [round(v, 10) for v in r.amplitude]
    [3.0]
    """
    ys = [float(v) for v in y]
    ts = [float(v) for v in t]
    X = []
    for tv in ts:
        row = [1.0] + ([tv] if trend else [])
        for k in range(1, n_harmonics + 1):
            row += [math.cos(2 * math.pi * k * tv / period), math.sin(2 * math.pi * k * tv / period)]
        X.append(row)
    p = len(X[0])
    G = [[ssum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)]
    beta = solve(G, [ssum(r[a] * v for r, v in zip(X, ys)) for a in range(p)])
    fitted = [ssum(r[a] * beta[a] for a in range(p)) for r in X]
    off = 2 if trend else 1
    amp = [math.hypot(beta[off + 2 * k], beta[off + 2 * k + 1]) for k in range(n_harmonics)]
    ph = [math.atan2(beta[off + 2 * k + 1], beta[off + 2 * k]) for k in range(n_harmonics)]
    my = ssum(ys) / len(ys)
    r2 = 1.0 - ssum((a - b) ** 2 for a, b in zip(ys, fitted)) / ssum((a - my) ** 2 for a in ys)
    return RichResult(payload={"coefficients": beta, "amplitude": amp, "phase": ph, "fitted": fitted, "r2": r2})


def cheatsheet() -> str:
    return (
        "cp_als / tucker_hooi / matrix_factorization_als / haar_dwt / haar_mra / haar_dwt_2d / haar_mra_2d / "
        "wavelet_detrend / harmonic_regression -> space-time decompositions."
    )


# alias kept from the retired placeholder of the same name
tensor_3way_sp = cp_als

# alias kept from the retired placeholder of the same name
tensor_decomp_sp = tucker_hooi

# alias kept from the retired placeholder of the same name
wavelet_mra_sp = haar_mra_2d

# alias kept from the retired placeholder of the same name
wavelet_spatial = haar_dwt_2d
