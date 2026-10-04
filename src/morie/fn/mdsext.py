# morie.fn -- function file (rootcoder007/morie)
"""Three-way SMACOF (INDSCAL, IDIOSCAL, replicated MDS), jackknife and bootstrap stability of an MDS
solution, oblique Procrustes target rotation, and reflection, flip, polarity and anisotropy checks of
a configuration."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .mdsops import _dist, _pairs, _sq, _torgerson, smacof

__all__ = [
    "smacof_indiff",
    "mds_jackknife",
    "mds_bootstrap",
    "procrustes_oblique",
    "mds_reflect",
    "mds_flip",
    "mds_polarity",
    "mds_anisotropy",
]


def _mat(X):
    return [[float(v) for v in r] for r in (X.tolist() if hasattr(X, "tolist") else X)]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _t(A):
    return [list(r) for r in zip(*A)]


def _inv(A):
    return [[float(v) for v in r] for r in inverse([list(r) for r in A])]


def _geninv(V):
    """``smacof:::myGenInv``: ``(V + 11'/n)^-1 - 11'/n``."""
    n = len(V)
    return [[v - 1.0 / n for v in r] for r in _inv([[v + 1.0 / n for v in r] for r in V])]


def _procrustus(M):
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    return _mm(_mat(U), _mat(Vt))


def _vmat(w, n, P):
    V = [[0.0] * n for _ in range(n)]
    for k, (i, j) in enumerate(P):
        V[i][j] = V[j][i] = -w[k]
    for i in range(n):
        V[i][i] = -ssum(V[i][j] for j in range(n) if j != i)
    return V


def smacof_indiff(
    deltas, ndim: int = 2, *, constraint: str = "indscal", itmax: int = 1000, eps: float = 1e-6
) -> RichResult:
    r"""Three-way SMACOF for several dissimilarity matrices, as ``smacof::smacofIndDiff`` (ratio MDS).

    Each source ``j`` gets the configuration ``X_j = Z C_j`` with the group
    space ``Z``: ``indscal`` (diagonal ``C_j``, the subject weights of
    Carroll and Chang 1970), ``idioscal`` (general ``C_j``) or ``identity``
    (``C_j = I``: replicated MDS, one configuration for all sources).  Every
    iteration takes the unconstrained Guttman transforms ``V_j^+ B(X_j) X_j``
    and projects them on the constraint (De Leeuw and Heiser 1980);
    dissimilarities are normalised to ``sum w delta^2 = n(n-1)/2`` per source,
    missing (``nan``) ones get weight 0. Starts from the Torgerson solution of
    the summed dissimilarities; stops when raw stress decreases by less than
    ``eps``.

    :return: ``conf`` (per-source configurations), ``gspace``, ``cweights``,
        ``stress`` (stress-1 over all sources), ``sps`` (stress per source,
        %), ``spp`` (stress per point, %), ``confdist``, ``niter``.

    References
    ----------
    Carroll, J. D. and Chang, J. J. (1970). Analysis of individual differences
    in multidimensional scaling via an N-way generalization of Eckart-Young
    decomposition. *Psychometrika*, 35(3), 283-319.
    De Leeuw, J. and Mair, P. (2009). Multidimensional scaling using
    majorization: SMACOF in R. *Journal of Statistical Software*, 31(3), 1-30.

    Examples
    --------
    >>> D1 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    >>> round(smacof_indiff([D1, D1], 1, constraint="identity").stress, 10)
    0.0
    """
    if constraint not in ("indscal", "idioscal", "identity"):
        raise ValueError("constraint must be indscal, idioscal or identity")
    Ds = [_sq(d) for d in deltas]
    m, n = len(Ds), len(Ds[0])
    if any(len(D) != n for D in Ds):
        raise ValueError("all dissimilarity matrices must have the same size")
    if ndim > n - 1:
        raise ValueError("ndim must be at most n - 1")
    p = ndim
    P = _pairs(n)
    nn = len(P)
    w = [[0.0 if D[i][j] != D[i][j] else 1.0 for i, j in P] for D in Ds]
    dl = [[0.0 if D[i][j] != D[i][j] else D[i][j] for i, j in P] for D in Ds]
    dh = []
    for j in range(m):
        s = math.sqrt(nn / ssum(w[j][k] * dl[j][k] ** 2 for k in range(nn)))
        dh.append([v * s for v in dl[j]])
    V = [_vmat(w[j], n, P) for j in range(m)]
    Vp = [_geninv(M) for M in V]
    S = [[0.0] * n for _ in range(n)]
    for k, (a, b) in enumerate(P):
        S[a][b] = S[b][a] = ssum(dl[j][k] for j in range(m))
    Z = _torgerson(S, p)
    C = [[[1.0 if a == b else 0.0 for b in range(p)] for a in range(p)] for _ in range(m)]
    d0 = _dist(Z)
    lb = ssum(w[j][k] * d0[k] * dh[j][k] for j in range(m) for k in range(nn)) / ssum(
        w[j][k] * d0[k] ** 2 for j in range(m) for k in range(nn)
    )
    Z = [[v * lb for v in r] for r in Z]
    X = [Z for _ in range(m)]
    d = [[v * lb for v in d0] for _ in range(m)]
    sold = ssum(w[j][k] * (dh[j][k] - d[j][k]) ** 2 for j in range(m) for k in range(nn))
    itel = 1
    while True:
        Y = []
        for j in range(m):
            Bm = [[0.0] * n for _ in range(n)]
            for k, (a, b) in enumerate(P):
                v = w[j][k] * dh[j][k] / d[j][k] if d[j][k] >= 1e-12 else 0.0
                Bm[a][b] = Bm[b][a] = -v
            for a in range(n):
                Bm[a][a] = -ssum(Bm[a][b] for b in range(n) if b != a)
            Y.append(_mm(Vp[j], _mm(Bm, X[j])))
        if constraint == "identity":
            zs = [[ssum(r[c] for r in rows) for c in range(p)] for rows in zip(*[_mm(V[j], Y[j]) for j in range(m)])]
            U = [[ssum(V[j][a][b] for j in range(m)) for b in range(n)] for a in range(n)]
            Z = _mm(_geninv(U), zs)
            Y = [Z for _ in range(m)]
        elif constraint == "indscal":
            aux0 = [[0.0] * p for _ in range(n)]
            for j in range(m):
                VY, VZ = _mm(V[j], Y[j]), _mm(V[j], Z)
                c = [
                    ssum(Z[i][s] * VY[i][s] for i in range(n)) / ssum(Z[i][s] * VZ[i][s] for i in range(n))
                    for s in range(p)
                ]
                C[j] = [[c[a] if a == b else 0.0 for b in range(p)] for a in range(p)]
                for i in range(n):
                    for s in range(p):
                        aux0[i][s] += VY[i][s] * c[s]
            for s in range(p):
                M = [[ssum(C[j][s][s] ** 2 * V[j][a][b] for j in range(m)) for b in range(n)] for a in range(n)]
                col = _mm(_geninv(M), [[aux0[i][s]] for i in range(n)])
                for i in range(n):
                    Z[i][s] = col[i][0]
            Y = [_mm(Z, C[j]) for j in range(m)]
        else:
            aux0 = [[0.0] * p for _ in range(n)]
            K = [[0.0] * (n * p) for _ in range(n * p)]
            for j in range(m):
                VY = _mm(V[j], Y[j])
                a1 = _mm(_t(Z), VY)
                a2 = _mm(_t(Z), _mm(V[j], Z))
                C[j] = _mm(_inv(a2), a1)
                CC = _mm(C[j], _t(C[j]))
                YB = _mm(VY, _t(C[j]))
                for i in range(n):
                    for s in range(p):
                        aux0[i][s] += YB[i][s]
                for s in range(p):
                    for t in range(p):
                        for a in range(n):
                            for b in range(n):
                                K[s * n + a][t * n + b] += CC[s][t] * V[j][a][b]
            for s in range(p):
                for a in range(n):
                    for b in range(n):
                        K[s * n + a][s * n + b] += 1.0 / n
            rhs = [aux0[i][s] for s in range(p) for i in range(n)]
            Ki = _inv(K)
            z = [ssum(Ki[r][c] * rhs[c] for c in range(n * p)) for r in range(n * p)]
            Z = [[z[s * n + i] for s in range(p)] for i in range(n)]
            Y = [_mm(Z, C[j]) for j in range(m)]
        e = [_dist(Yj) for Yj in Y]
        snon = ssum(w[j][k] * (dh[j][k] - e[j][k]) ** 2 for j in range(m) for k in range(nn))
        if sold - snon < eps or itel == itmax:
            break
        X, d, sold = Y, e, snon
        itel += 1
    confdist, spps, sps = [], [], []
    for j in range(m):
        s = math.sqrt(nn / ssum(w[j][k] * e[j][k] ** 2 for k in range(nn)))
        cd = [v * s for v in e[j]]
        confdist.append(cd)
        R = [[0.0] * n for _ in range(n)]
        for k, (a, b) in enumerate(P):
            R[a][b] = R[b][a] = w[j][k] * (dh[j][k] - cd[k]) ** 2
        cm = [ssum(R[a][b] for a in range(n) if a != b) / (n - 1) for b in range(n)]
        tot = ssum(cm)
        spps.append([100.0 * v / tot for v in cm])
        sps.append(ssum(w[j][k] * (dh[j][k] - e[j][k]) ** 2 for k in range(nn)))
    tsps = ssum(sps)
    return RichResult(
        payload={
            "conf": Y,
            "gspace": Z,
            "cweights": C,
            "stress": math.sqrt(snon / m / nn),
            "sps": [100.0 * v / tsps for v in sps],
            "spp": [ssum(r[b] for r in spps) / m for b in range(n)],
            "confdist": confdist,
            "niter": itel,
            "constraint": constraint,
        }
    )


def _norm2(M, how):
    if how == "smacof":  # base::norm default type "O": maximum absolute column sum
        return max(ssum(abs(r[c]) for r in M) for c in range(len(M[0]))) ** 2
    return ssum(v * v for r in M for v in r)


def mds_jackknife(
    delta, ndim: int = 2, *, type: str = "ratio", method: str = "standard", eps: float = 1e-6, itmax: int = 100
) -> RichResult:
    r"""Jackknife stability of a SMACOF solution (De Leeuw and Meulman 1986), as ``smacof::jackmds``.

    Refits the MDS with each object left out (its row set to zero), rotates
    the ``n`` leave-one-out configurations ``X_i K_i`` to a common comparison
    configuration ``Y0`` (orthogonal Procrustes, alternating), and reports
    ``stab = 1 - sum ||Y_i - Y0||^2 / sum ||Y_i||^2``, ``cross = 1 - n ||X0 -
    Y0||^2 / sum ||Y_i||^2`` and ``disp = 2 - (stab + cross)``.
    ``method="standard"`` uses the full sum ``sum_j X_j K_j`` in the Procrustes
    update of ``K_i`` and Frobenius norms; ``method="smacof"`` reproduces
    ``jackmds`` 2.1, whose update sums over ``j >= i`` only and whose norms
    are ``base::norm``'s default maximum absolute column sum.

    References
    ----------
    De Leeuw, J. and Meulman, J. (1986). A special jackknife for
    multidimensional scaling. *Journal of Classification*, 3(1), 97-112.

    Examples
    --------
    >>> D = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
    >>> r = mds_jackknife(D)
    >>> 0 < r.stab <= 1 and abs(r.disp - (2 - r.stab - r.cross)) < 1e-15
    True
    """
    if method not in ("standard", "smacof"):
        raise ValueError("method must be standard or smacof")
    D = _sq(delta)
    n = len(D)
    x0 = smacof(D, ndim, type=type).conf
    xx = []
    for i in range(n):
        sub = [[D[a][b] for b in range(n) if b != i] for a in range(n) if a != i]
        rows = iter(smacof(sub, ndim, type=type).conf)
        xx.append([[0.0] * ndim if a == i else [float(v) for v in next(rows)] for a in range(n)])
    eye = [[1.0 if a == b else 0.0 for b in range(ndim)] for a in range(ndim)]
    K = [eye for _ in range(n)]
    oloss, itel = math.inf, 1
    while True:
        XK = [_mm(xx[i], K[i]) for i in range(n)]
        y0 = [[ssum(XK[i][a][c] for i in range(n)) * (n - 1) / (n * (n - 2)) for c in range(ndim)] for a in range(n)]
        for i in range(n):
            js = range(i, n) if method == "smacof" else range(n)
            zz = [[ssum(_mm(xx[j], K[j])[a][c] for j in js) for c in range(ndim)] for a in range(n)]
            K[i] = _procrustus(_mm(_t(xx[i]), zz))
        yy, nloss = [], 0.0
        for i in range(n):
            Yi = _mm(xx[i], K[i])
            Yi[i] = [n * v / (n - 1) for v in y0[i]]
            Yi = [[Yi[a][c] - y0[i][c] / (n - 1) for c in range(ndim)] for a in range(n)]
            yy.append(Yi)
            nloss += ssum((y0[a][c] - Yi[a][c]) ** 2 for a in range(n) for c in range(ndim))
        if oloss - nloss < eps or itel == itmax:
            break
        itel += 1
        oloss = nloss
    x0 = _mm(x0, _procrustus(_mm(_t(x0), y0)))
    den = ssum(_norm2(Y, method) for Y in yy)
    stab = 1.0 - ssum(_norm2([[Y[a][c] - y0[a][c] for c in range(ndim)] for a in range(n)], method) for Y in yy) / den
    cross = 1.0 - n * _norm2([[x0[a][c] - y0[a][c] for c in range(ndim)] for a in range(n)], method) / den
    return RichResult(
        payload={
            "smacof_conf": x0,
            "jackknife_conf": yy,
            "comparison_conf": y0,
            "stab": stab,
            "cross": cross,
            "disp": 2.0 - (stab + cross),
            "niter": itel,
            "loss": nloss,
        }
    )


def _ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _diss(rows, how):
    cols = [list(c) for c in zip(*rows)]
    n = len(cols)
    if how == "euclidean":
        return [[math.sqrt(ssum((a - b) ** 2 for a, b in zip(cols[i], cols[j]))) for j in range(n)] for i in range(n)]
    if how == "spearman":
        cols = [_ranks(c) for c in cols]
    elif how != "pearson":
        raise ValueError("method_dat must be pearson, spearman or euclidean")
    N = len(cols[0])
    cen = []
    for c in cols:
        mu = ssum(c) / N
        cen.append([v - mu for v in c])
    sd = [math.sqrt(ssum(v * v for v in c)) for c in cen]
    return [
        [
            0.0 if i == j else math.sqrt(max(0.0, 1.0 - ssum(a * b for a, b in zip(cen[i], cen[j])) / (sd[i] * sd[j])))
            for j in range(n)
        ]
        for i in range(n)
    ]


def _quantile7(x, q):
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def mds_bootstrap(
    data,
    ndim: int = 2,
    *,
    method_dat: str = "pearson",
    nrep: int = 100,
    alpha: float = 0.05,
    type: str = "ratio",
    method: str = "standard",
    seed: int = 1,
    resamples=None,
) -> RichResult:
    r"""Bootstrap confidence for a SMACOF solution of variables (Jacoby and Armstrong 2014), as ``smacof::bootmds``.

    ``data`` is an ``N x n`` matrix whose ``n`` columns are the MDS objects.
    Dissimilarities are ``sqrt(1 - r)`` from Pearson or Spearman
    correlations, or Euclidean distances between columns. Each replicate
    resamples rows with replacement (Philox stream ``r`` of ``seed``, or the
    0-based index lists in ``resamples``), refits the MDS and Procrustes-fits
    it (rotation, dilation, translation) onto the original configuration.
    Returns per-object covariance matrices of the bootstrap coordinates
    (confidence ellipses), the stress percentile interval and the stability
    ``1 - sum ||Y_r - Ybar||^2 / sum ||Y_r||^2`` (``method="smacof"``: with
    ``base::norm``'s default maximum absolute column sum, as ``bootmds``).

    References
    ----------
    Jacoby, W. G. and Armstrong, D. A. (2014). Bootstrap confidence regions
    for multidimensional scaling solutions. *American Journal of Political
    Science*, 58(1), 264-278.

    Examples
    --------
    >>> X = [[1, 2, 3, 1], [2, 1, 4, 0], [3, 5, 2, 2], [4, 3, 6, 1], [5, 6, 5, 3], [6, 4, 8, 2]]
    >>> r = mds_bootstrap(X, 2, method_dat="euclidean", nrep=5)
    >>> len(r.cov), r.nrep
    (4, 5)
    """
    if method not in ("standard", "smacof"):
        raise ValueError("method must be standard or smacof")
    rows = _mat(data)
    N, n = len(rows), len(rows[0])
    fit0 = smacof(_diss(rows, method_dat), ndim, type=type)
    X0 = fit0.conf
    mx = [ssum(r[c] for r in X0) / n for c in range(ndim)]
    Xc = [[r[c] - mx[c] for c in range(ndim)] for r in X0]
    if resamples is None:
        resamples = [[min(N - 1, int(u * N)) for u in random_uniform(N, seed=seed, stream=r)] for r in range(nrep)]
    coord, stressvec = [], []
    for idx in resamples:
        o = smacof(_diss([rows[i] for i in idx], method_dat), ndim, type=type)
        stressvec.append(o.stress)
        Y = o.conf
        my = [ssum(r[c] for r in Y) / n for c in range(ndim)]
        Yc = [[r[c] - my[c] for c in range(ndim)] for r in Y]
        T = _procrustus(_mm(_t(Xc), Yc))
        T = _t(T)  # T = Q P' for svd(X'ZY) = P D Q'
        YT = _mm(Yc, T)
        c = ssum(Xc[i][k] * YT[i][k] for i in range(n) for k in range(ndim)) / ssum(v * v for r in Yc for v in r)
        coord.append([[c * YT[i][k] + mx[k] for k in range(ndim)] for i in range(n)])
    R = len(coord)
    cov = []
    for k in range(n):
        pts = [cd[k] for cd in coord]
        mu = [ssum(p[a] for p in pts) / R for a in range(ndim)]
        cov.append(
            [[ssum((p[a] - mu[a]) * (p[b] - mu[b]) for p in pts) / (R - 1) for b in range(ndim)] for a in range(ndim)]
        )
    y0 = [[ssum(cd[i][k] for cd in coord) / R for k in range(ndim)] for i in range(n)]
    stab = 1.0 - (
        ssum(_norm2([[cd[i][k] - y0[i][k] for k in range(ndim)] for i in range(n)], method) for cd in coord)
        / ssum(_norm2(cd, method) for cd in coord)
    )
    return RichResult(
        payload={
            "conf": X0,
            "bootconf": coord,
            "cov": cov,
            "stressvec": stressvec,
            "bootci": [_quantile7(stressvec, alpha / 2), _quantile7(stressvec, 1 - alpha / 2)],
            "stab": stab,
            "nrep": R,
            "stress": fit0.stress,
        }
    )


def procrustes_oblique(A, target, *, eps: float = 1e-8, maxit: int = 5000, fwindow: int = 10) -> RichResult:
    r"""Oblique Procrustes (target) rotation ``L = A (T')^{-1}`` minimising ``||L - B||^2`` with unit-length columns of ``T``.

    Browne's (1967, 2001) oblique target rotation by the gradient-projection
    algorithm of Jennrich (2002) (``GPArotation::GPFoblq`` with the target
    criterion, default ``algorithm = "bb"``): step ``T - alpha G_p`` projected
    back to unit columns, ``alpha`` the Barzilai-Borwein step clipped to
    ``[1e-10, 20]`` (doubled on the first iteration) and halved until a
    non-monotone Armijo condition against the largest criterion of the last
    ``fwindow`` iterations holds. ``nan`` target entries are unspecified.
    Returns the rotated ``loadings``, ``Phi = T'T`` (axis correlations), ``T``
    and the criterion ``f``.

    References
    ----------
    Browne, M. W. (2001). An overview of analytic rotation in exploratory
    factor analysis. *Multivariate Behavioral Research*, 36(1), 111-150.
    Jennrich, R. I. (2002). A simple general method for oblique rotation.
    *Psychometrika*, 67(1), 7-19.

    Examples
    --------
    >>> A = [[1, 0], [0, 1], [1, 1]]
    >>> round(procrustes_oblique(A, A).f, 12)
    0.0
    """
    Am = _mat(A)
    B = _mat(target)
    k = len(Am[0])

    def crit(Tm):
        Ti = _inv(Tm)
        L = _mm(Am, _t(Ti))
        f = ssum((L[i][c] - B[i][c]) ** 2 for i in range(len(L)) for c in range(k) if B[i][c] == B[i][c])
        Gq = [[2.0 * (L[i][c] - B[i][c]) if B[i][c] == B[i][c] else 0.0 for c in range(k)] for i in range(len(L))]
        G = [[-v for v in r] for r in _t(_mm(_mm(_t(L), Gq), Ti))]
        return L, f, G

    T = [[1.0 if a == b else 0.0 for b in range(k)] for a in range(k)]
    L, f, G = crit(T)
    alpha = 1.0
    fs: list[float] = []
    T_prev = Gp_prev = None
    it = 0
    while True:
        cs = [ssum(T[a][c] * G[a][c] for a in range(k)) for c in range(k)]
        Gp = [[G[a][c] - T[a][c] * cs[c] for c in range(k)] for a in range(k)]
        s = math.sqrt(ssum(v * v for r in Gp for v in r))
        fs.append(f)
        if s < eps or it == maxit + 1:
            break
        if T_prev is not None:
            dT = [T[a][c] - T_prev[a][c] for a in range(k) for c in range(k)]
            dG = [Gp[a][c] - Gp_prev[a][c] for a in range(k) for c in range(k)]
            if ssum(v * v for v in dG) > 0:
                alpha = max(1e-10, min(ssum(v * v for v in dT) / abs(ssum(a * b for a, b in zip(dT, dG))), 20.0))
        else:
            alpha *= 2.0
        target_f = max(fs[-fwindow:])
        for _ in range(11):
            Xm = [[T[a][c] - alpha * Gp[a][c] for c in range(k)] for a in range(k)]
            v = [1.0 / math.sqrt(ssum(Xm[a][c] ** 2 for a in range(k))) for c in range(k)]
            Tt = [[Xm[a][c] * v[c] for c in range(k)] for a in range(k)]
            Lt, ft, Gt = crit(Tt)
            if target_f - ft > 0.5 * s * s * alpha:
                break
            alpha /= 2.0
        T_prev, Gp_prev = T, Gp
        T, L, f, G = Tt, Lt, ft, Gt
        it += 1
    return RichResult(
        payload={"loadings": L, "Phi": _mm(_t(T), T), "T": T, "f": f, "iterations": it, "converged": s < eps}
    )


def mds_reflect(X, target=None) -> RichResult:
    r"""Reflect configuration axes: to agree in sign with ``target`` (``sum_i x_ik t_ik >= 0``) or, without one, so the largest-magnitude coordinate of each axis is positive.

    MDS solutions are identified only up to rotation and reflection; this
    fixes the reflection so solutions from different programs, starts or
    samples can be compared (Borg and Groenen 2005, section 7.10).

    References
    ----------
    Borg, I. and Groenen, P. J. F. (2005). *Modern Multidimensional Scaling*
    (2nd ed.). Springer.

    Examples
    --------
    >>> mds_reflect([[1, -3], [-2, 1]]).signs
    [-1, -1]
    """
    M = _mat(X)
    k = len(M[0])
    signs = []
    for c in range(k):
        if target is not None:
            T = _mat(target)
            sgn = -1 if ssum(M[i][c] * T[i][c] for i in range(len(M))) < 0 else 1
        else:
            big = max(range(len(M)), key=lambda i: (abs(M[i][c]), -i))
            sgn = -1 if M[big][c] < 0 else 1
        signs.append(sgn)
    return RichResult(payload={"conf": [[r[c] * signs[c] for c in range(k)] for r in M], "signs": signs})


def mds_flip(X, Y) -> RichResult:
    r"""Does configuration ``Y`` match ``X`` only after a reflection? Orthogonal Procrustes check.

    Centres both, fits the orthogonal rotation ``R = V U'`` (``X'Y = U S
    V'``) of ``Y`` onto ``X``; ``det(R) < 0`` means an improper rotation
    (a flip). Also returns the per-axis correlations of the raw coordinates,
    the axes whose correlation is negative (``flipped_axes``) and the fit
    ``||X - Y R||^2`` before and after.

    References
    ----------
    Gower, J. C. and Dijksterhuis, G. B. (2004). *Procrustes Problems*.
    Oxford University Press.

    Examples
    --------
    >>> r = mds_flip([[0, 0], [1, 0], [0, 2]], [[0, 0], [-1, 0], [0, 2]])
    >>> r.reflected, r.flipped_axes
    (True, [0])
    """
    Xm, Ym = _mat(X), _mat(Y)
    n, k = len(Xm), len(Xm[0])

    def cen(M):
        mu = [ssum(r[c] for r in M) / n for c in range(k)]
        return [[r[c] - mu[c] for c in range(k)] for r in M]

    Xc, Yc = cen(Xm), cen(Ym)
    U, _, Vt = np.linalg.svd(np.asarray(_mm(_t(Xc), Yc), dtype=float))
    R = _mm(_t(_mat(Vt)), _t(_mat(U)))
    det = float(np.linalg.det(np.asarray(R, dtype=float)))
    cor = []
    for c in range(k):
        sx = math.sqrt(ssum(r[c] ** 2 for r in Xc))
        sy = math.sqrt(ssum(r[c] ** 2 for r in Yc))
        cor.append(ssum(Xc[i][c] * Yc[i][c] for i in range(n)) / (sx * sy) if sx > 0 and sy > 0 else 0.0)
    YR = _mm(Yc, R)
    return RichResult(
        payload={
            "rotation": R,
            "determinant": det,
            "reflected": det < 0,
            "axis_correlations": cor,
            "flipped_axes": [c for c in range(k) if cor[c] < 0],
            "ss_before": ssum((Xc[i][c] - Yc[i][c]) ** 2 for i in range(n) for c in range(k)),
            "ss_after": ssum((Xc[i][c] - YR[i][c]) ** 2 for i in range(n) for c in range(k)),
        }
    )


def mds_polarity(X, anchors) -> RichResult:
    r"""Orient each dimension so its anchor object has a non-negative coordinate (the ``polarity`` convention of W-NOMINATE).

    ``anchors[k]`` is the (0-based) row that must lie on the positive side of
    dimension ``k`` (e.g. a known conservative legislator). Returns the
    oriented ``conf``, the applied ``signs`` and the ``poles`` of every
    dimension (rows with the minimum and maximum coordinate).

    References
    ----------
    Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*. Cambridge
    University Press.

    Examples
    --------
    >>> mds_polarity([[1, 2], [-1, -2], [0, 0]], [1, 0]).signs
    [-1, 1]
    """
    M = _mat(X)
    k = len(M[0])
    if len(anchors) != k:
        raise ValueError("one anchor per dimension")
    signs = [-1 if M[anchors[c]][c] < 0 else 1 for c in range(k)]
    C = [[r[c] * signs[c] for c in range(k)] for r in M]
    poles = [[min(range(len(C)), key=lambda i: C[i][c]), max(range(len(C)), key=lambda i: C[i][c])] for c in range(k)]
    return RichResult(payload={"conf": C, "signs": signs, "poles": poles})


def mds_anisotropy(X) -> RichResult:
    r"""Anisotropy of a configuration: eigenvalues of its covariance, ratio ``lambda_1 / lambda_p`` and principal-axis angle.

    ``ratio`` 1 is an isotropic (round) cloud; large values mean the objects
    spread mainly along one direction. ``angle`` (2-D and up) is the angle in
    degrees, in ``[0, 180)``, of the first principal axis from dimension 1 in
    the plane of the first two dimensions; ``eccentricity`` is ``sqrt(1 -
    lambda_p / lambda_1)``.

    References
    ----------
    Borg, I. and Groenen, P. J. F. (2005). *Modern Multidimensional Scaling*
    (2nd ed.). Springer.

    Examples
    --------
    >>> r = mds_anisotropy([[-2, 0], [2, 0], [0, -1], [0, 1]])
    >>> round(r.ratio, 12), round(r.angle, 12)
    (4.0, 0.0)
    """
    M = _mat(X)
    n, k = len(M), len(M[0])
    mu = [ssum(r[c] for r in M) / n for c in range(k)]
    Cv = [[ssum((r[a] - mu[a]) * (r[b] - mu[b]) for r in M) / (n - 1) for b in range(k)] for a in range(k)]
    w, V = np.linalg.eigh(np.asarray(Cv, dtype=float))
    w = [float(v) for v in w.tolist()][::-1]
    V = _mat(V)
    v1 = [V[a][k - 1] for a in range(k)]
    ang = math.degrees(math.atan2(v1[1], v1[0])) % 180.0 if k > 1 else 0.0
    if ang >= 180.0 - 1e-12:
        ang = 0.0
    return RichResult(
        payload={
            "eigenvalues": w,
            "ratio": w[0] / w[-1] if w[-1] > 0 else math.inf,
            "angle": ang,
            "eccentricity": math.sqrt(max(0.0, 1.0 - w[-1] / w[0])) if w[0] > 0 else 0.0,
        }
    )


def cheatsheet() -> str:
    return (
        "smacof_indiff / mds_jackknife / mds_bootstrap / procrustes_oblique / mds_reflect / mds_flip / "
        "mds_polarity / mds_anisotropy -> three-way MDS, stability and orientation checks."
    )
