# morie.fn -- function file (rootcoder007/morie)
"""SMACOF multidimensional scaling, Procrustes analysis and embedding-quality measures."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = [
    "smacof",
    "procrustes_fit",
    "generalized_procrustes",
    "embedding_quality",
    "alienation_coefficient",
    "dissimilarity_check",
]


def _sq(M):
    A = [[float(v) for v in r] for r in np.asarray(M, dtype=float).tolist()]
    n = len(A)
    if any(len(r) != n for r in A):
        raise ValueError("expected a square matrix")
    return A


def _pairs(n):
    """Lower-triangle pairs in R ``dist`` order (column-major)."""
    return [(i, j) for j in range(n) for i in range(j + 1, n)]


def _dist(X):
    n = len(X)
    return [math.sqrt(ssum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[0])))) for i, j in _pairs(n)]


def _torgerson(D, p):
    n = len(D)
    D2 = [[v * v for v in r] for r in D]
    rm = [ssum(r) / n for r in D2]
    gm = ssum(rm) / n
    B = [[-0.5 * (D2[i][j] - rm[i] - rm[j] + gm) for j in range(n)] for i in range(n)]
    vals, vecs = np.linalg.eigh(np.asarray(B, dtype=float))
    vals = [float(v) for v in vals.tolist()]
    V = vecs.tolist()
    order = sorted(range(n), key=lambda k: -vals[k])[:p]
    return [[float(V[i][k]) * math.sqrt(max(vals[k], 0.0)) for k in order] for i in range(n)]


def _pava(y, w):
    vals, wts, sizes = [], [], []
    for v, u in zip(y, w):
        vals.append(v)
        wts.append(u)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw if tw > 0 else 0.5 * (vals[-2] + vals[-1])
            wts[-2] = tw
            sizes[-2] += sizes[-1]
            del vals[-1], wts[-1], sizes[-1]
    return [v for v, k in zip(vals, sizes) for _ in range(k)]


def smacof(
    delta, ndim: int = 2, *, type: str = "ratio", weights=None, init=None, itmax: int = 1000, eps: float = 1e-6
) -> RichResult:
    r"""Symmetric SMACOF multidimensional scaling, as ``smacof::smacofSym`` (De Leeuw and Mair 2009).

    Minimises raw stress ``sum w_ij (dhat_ij - d_ij(X))^2 / (n(n-1)/2)`` by
    the Guttman transform ``X <- V^+ B(X) X`` from the Torgerson start
    (rescaled by ``sum w d dhat / sum w d^2``), alternating with the optimal
    scaling of the disparities ``dhat`` normalised to ``sum w dhat^2 =
    n(n-1)/2``: ``ratio`` (proportional to ``delta``), ``interval`` (a
    weighted least-squares line in ``delta`` over its tie blocks) or
    ``ordinal`` (weighted monotone regression on the order of ``(delta,
    d)``, Kruskal's primary treatment of ties).  Iteration stops when the
    decrease of raw stress falls below ``eps``.  ``weights`` 0 (or ``nan``
    in ``delta``) marks missing dissimilarities.  Configurations agree with
    ``smacof`` up to reflections of the axes (the eigenvector signs of the
    Torgerson start); distances and stress agree exactly.

    :return: :class:`RichResult` with ``conf``, ``stress`` (square root of
        normalised raw stress, stress-1), ``dhat`` and ``confdist`` (pairs in
        R ``dist`` order), ``spp`` (stress per point, %), ``niter``.

    References
    ----------
    De Leeuw, J. and Mair, P. (2009). Multidimensional scaling using
    majorization: SMACOF in R. *Journal of Statistical Software*, 31(3),
    1-30.
    Kruskal, J. B. (1964). Nonmetric multidimensional scaling: a numerical
    method. *Psychometrika*, 29(2), 115-129.

    Examples
    --------
    >>> D = [[0, 3, 4, 6], [3, 0, 5, 4], [4, 5, 0, 3], [6, 4, 3, 0]]
    >>> round(smacof(D).stress, 6)
    0.049612
    """
    if type not in ("ratio", "interval", "ordinal"):
        raise ValueError("type must be ratio, interval or ordinal")
    D = _sq(delta)
    n = len(D)
    if ndim > n - 1:
        raise ValueError("ndim must be at most n - 1")
    W = [[1.0] * n for _ in range(n)] if weights is None else _sq(weights)
    P = _pairs(n)
    nn = len(P)
    dl = [D[i][j] for i, j in P]
    w = [0.0 if dl[k] != dl[k] else W[P[k][0]][P[k][1]] for k in range(nn)]
    obs = [k for k in range(nn) if dl[k] == dl[k]]
    mean_d = ssum(dl[k] for k in obs) / len(obs)
    Df = [[(D[i][j] if D[i][j] == D[i][j] else mean_d) for j in range(n)] for i in range(n)]
    X = [list(r) for r in (np.asarray(init, dtype=float).tolist() if init is not None else _torgerson(Df, ndim))]
    s0 = math.sqrt(nn / ssum(w[k] * dl[k] ** 2 for k in obs))
    dhat = [dl[k] * s0 if dl[k] == dl[k] else 1.0 for k in range(nn)]
    # V^+ = (V + 11'/n)^-1 - 11'/n
    Wm = [[0.0] * n for _ in range(n)]
    for k, (i, j) in enumerate(P):
        Wm[i][j] = Wm[j][i] = w[k]
    V = [[(ssum(Wm[i]) if i == j else -Wm[i][j]) + 1.0 / n for j in range(n)] for i in range(n)]
    Vp = [[v - 1.0 / n for v in r] for r in inverse(V)]
    Vp = [[float(v) for v in r] for r in Vp]
    # optimal-scaling set-up: tie blocks of the observed dissimilarities
    iord = sorted(obs, key=lambda k: (dl[k], k))
    blocks = []
    for k in iord:
        if blocks and dl[blocks[-1][-1]] == dl[k]:
            blocks[-1].append(k)
        else:
            blocks.append([k])

    def transform(e):
        R = [0.0] * nn
        if type == "ratio":
            for k in obs:
                R[k] = dl[k]
        elif type == "interval":
            yb, wb, xb = [], [], []
            for b in blocks:
                sw = ssum(w[k] for k in b)
                yb.append(ssum(w[k] * e[k] for k in b) / sw if sw > 0 else 0.0)
                wb.append(sw)
                xb.append(dl[b[0]] - dl[blocks[0][0]])
            a11, a12, a22 = ssum(wb), ssum(u * x for u, x in zip(wb, xb)), ssum(u * x * x for u, x in zip(wb, xb))
            f1, f2 = ssum(u * y for u, y in zip(wb, yb)), ssum(u * x * y for u, x, y in zip(wb, xb, yb))
            det = a11 * a22 - a12 * a12
            b0, b1 = (a22 * f1 - a12 * f2) / det, (a11 * f2 - a12 * f1) / det
            for b, x in zip(blocks, xb):
                for k in b:
                    R[k] = b0 + b1 * x
        else:
            order = sorted(obs, key=lambda k: (dl[k], e[k], k))
            fit = _pava([e[k] for k in order], [w[k] for k in order])
            for k, v in zip(order, fit):
                R[k] = v
        s = math.sqrt(nn / ssum(w[k] * R[k] ** 2 for k in range(nn)))
        return [v * s for v in R]

    d = _dist(X)
    lb = ssum(w[k] * d[k] * dhat[k] for k in range(nn)) / ssum(w[k] * d[k] ** 2 for k in range(nn))
    X = [[v * lb for v in r] for r in X]
    d = [v * lb for v in d]
    sold = ssum(w[k] * (dhat[k] - d[k]) ** 2 for k in range(nn)) / nn
    itel = 1
    while True:
        Bm = [[0.0] * n for _ in range(n)]
        for k, (i, j) in enumerate(P):
            b = w[k] * dhat[k] / d[k] if d[k] >= 1e-12 else 0.0
            Bm[i][j] = Bm[j][i] = -b
        for i in range(n):
            Bm[i][i] = -ssum(Bm[i][j] for j in range(n) if j != i)
        BX = [[ssum(Bm[i][t] * X[t][c] for t in range(n)) for c in range(ndim)] for i in range(n)]
        Y = [[ssum(Vp[i][t] * BX[t][c] for t in range(n)) for c in range(ndim)] for i in range(n)]
        e = _dist(Y)
        dhat = transform(e)
        snon = ssum(w[k] * (dhat[k] - e[k]) ** 2 for k in range(nn)) / nn
        if sold - snon < eps or itel == itmax:
            break
        X, d, sold = Y, e, snon
        itel += 1
    res = [[0.0] * n for _ in range(n)]
    for k, (i, j) in enumerate(P):
        res[i][j] = res[j][i] = w[k] * (dhat[k] - e[k]) ** 2
    cm = [ssum(res[i][j] for i in range(n) if i != j) / (n - 1) for j in range(n)]
    tot = ssum(cm)
    return RichResult(
        payload={
            "conf": Y,
            "stress": math.sqrt(snon),
            "dhat": [dhat[k] if dl[k] == dl[k] else float("nan") for k in range(nn)],
            "confdist": e,
            "spp": [100.0 * v / tot for v in cm],
            "niter": itel,
            "weights": w,
        }
    )


def procrustes_fit(X, Y, *, scale: bool = True, symmetric: bool = False) -> RichResult:
    r"""Procrustes rotation of ``Y`` onto the target ``X``, as ``vegan::procrustes`` / ``protest``.

    Both configurations are centred (and, with ``symmetric``, scaled to unit
    sum of squares); with the SVD ``X'Y = U S V'`` the rotation is ``A = V
    U'``, the dilation ``c = sum(S) / tr(Y'Y)`` (1 without ``scale``), the
    fitted ``Yrot = c Y A`` and ``ss = tr(X'X) + c^2 tr(Y'Y) - 2 c sum(S)``.
    Also returned: per-row ``residuals`` ``||x_i - yrot_i||``, the
    translation and the Procrustes correlation ``r = sqrt(1 - ss_sym)`` of
    ``protest`` (Gower 1971; Peres-Neto and Jackson 2001).

    References
    ----------
    Gower, J. C. (1971). Statistical methods of comparing different
    multivariate analyses of the same data. In *Mathematics in the
    Archaeological and Historical Sciences*, 138-149. Edinburgh University
    Press.
    Peres-Neto, P. R. and Jackson, D. A. (2001). How well do multivariate
    data sets match? *Oecologia*, 129(2), 169-178.

    Examples
    --------
    >>> r = procrustes_fit([[0, 0], [1, 0], [0, 1]], [[0, 0], [0, 2], [-2, 0]])
    >>> round(r.ss, 12), round(r.scale, 6), round(r.correlation, 6)
    (0.0, 0.5, 1.0)
    """
    Xm = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    Ym = [[float(v) for v in r] for r in np.asarray(Y, dtype=float).tolist()]
    n = len(Xm)
    if len(Ym) != n:
        raise ValueError("X and Y must have the same number of rows")
    k = max(len(Xm[0]), len(Ym[0]))
    Xm = [r + [0.0] * (k - len(r)) for r in Xm]
    Ym = [r + [0.0] * (k - len(r)) for r in Ym]

    def centre(M):
        mu = [ssum(M[i][c] for i in range(n)) / n for c in range(k)]
        return [[M[i][c] - mu[c] for c in range(k)] for i in range(n)], mu

    def fit(A, B, sc):
        XY = [[ssum(A[i][a] * B[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        U, S, Vt = np.linalg.svd(np.asarray(XY, dtype=float))
        U, S, Vt = U.tolist(), [float(v) for v in S.tolist()], Vt.tolist()
        R = [[ssum(Vt[t][a] * U[b][t] for t in range(k)) for b in range(k)] for a in range(k)]  # A = V U'
        trB = ssum(v * v for r in B for v in r)
        trA = ssum(v * v for r in A for v in r)
        c = ssum(S) / trB if sc else 1.0
        Yr = [[c * ssum(B[i][t] * R[t][b] for t in range(k)) for b in range(k)] for i in range(n)]
        return R, c, Yr, trA + c * c * trB - 2.0 * c * ssum(S), S

    Xc, mx = centre(Xm)
    Yc, my = centre(Ym)
    if symmetric:
        sx = math.sqrt(ssum(v * v for r in Xc for v in r))
        sy = math.sqrt(ssum(v * v for r in Yc for v in r))
        Xc = [[v / sx for v in r] for r in Xc]
        Yc = [[v / sy for v in r] for r in Yc]
    R, c, Yr, ss, S = fit(Xc, Yc, scale)
    sx = math.sqrt(ssum(v * v for r in Xc for v in r))
    sy = math.sqrt(ssum(v * v for r in Yc for v in r))
    Xs = [[v / sx for v in r] for r in Xc]
    Ys = [[v / sy for v in r] for r in Yc]
    ss_sym = fit(Xs, Ys, True)[3]
    trans = [mx[b] - c * ssum(my[t] * R[t][b] for t in range(k)) for b in range(k)]
    resid = [math.sqrt(ssum((Xc[i][b] - Yr[i][b]) ** 2 for b in range(k))) for i in range(n)]
    return RichResult(
        payload={
            "Yrot": Yr,
            "rotation": R,
            "scale": c,
            "translation": trans,
            "ss": ss,
            "residuals": resid,
            "correlation": math.sqrt(max(0.0, 1.0 - ss_sym)),
        }
    )


def generalized_procrustes(configs, *, tol: float = 1e-12, maxit: int = 1000) -> RichResult:
    r"""Generalised orthogonal Procrustes analysis (Gower 1975), rotations and translations.

    Each configuration is centred; each is then repeatedly rotated (orthogonal
    Procrustes, no dilation) onto the mean of the others' current fits until
    the total residual sum of squares ``sum_i ||X_i R_i - G||^2`` about the
    consensus ``G`` decreases by less than ``tol``.

    References
    ----------
    Gower, J. C. (1975). Generalized Procrustes analysis. *Psychometrika*,
    40(1), 33-51.

    Examples
    --------
    >>> A = [[0, 0], [2, 0], [0, 1]]
    >>> B = [[0, 0], [0, 2], [-1, 0]]
    >>> round(generalized_procrustes([A, B]).rss, 10)
    0.0
    """
    Ms = [[[float(v) for v in r] for r in np.asarray(C, dtype=float).tolist()] for C in configs]
    m = len(Ms)
    n, k = len(Ms[0]), len(Ms[0][0])
    if m < 2 or any(len(M) != n or len(M[0]) != k for M in Ms):
        raise ValueError("need at least two configurations of equal shape")
    cur = []
    for M in Ms:
        mu = [ssum(M[i][c] for i in range(n)) / n for c in range(k)]
        cur.append([[M[i][c] - mu[c] for c in range(k)] for i in range(n)])
    orig = [r for r in cur]

    def consensus(F):
        return [[ssum(F[t][i][c] for t in range(m)) / m for c in range(k)] for i in range(n)]

    def rss(F, G):
        return ssum((F[t][i][c] - G[i][c]) ** 2 for t in range(m) for i in range(n) for c in range(k))

    G = consensus(cur)
    old = rss(cur, G)
    sweeps = 0
    for _ in range(maxit):
        sweeps += 1
        for t in range(m):
            # the consensus is refreshed after every rotation (Gower's sequential algorithm)
            others = [[(G[i][c] * m - cur[t][i][c]) / (m - 1) for c in range(k)] for i in range(n)]
            f = procrustes_fit(others, orig[t], scale=False)
            cur[t] = f["Yrot"]
            G = consensus(cur)
        new = rss(cur, G)
        if old - new < tol:
            old = new
            break
        old = new
    return RichResult(payload={"fitted": cur, "consensus": G, "rss": old, "iterations": sweeps})


def _ranks(D):
    n = len(D)
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (D[i][j], j))
        for r, j in enumerate(order, start=1):
            R[i][j] = r
    return R


def embedding_quality(D_high, D_low, k: int) -> RichResult:
    r"""Neighbourhood-preservation measures of an embedding at neighbourhood size ``k``.

    From the neighbour ranks ``rho_ij`` (high-dimensional) and ``r_ij``
    (embedding), ties broken by index: trustworthiness ``1 - 2/(n k (2n - 3k
    - 1)) sum_i sum_{j in U_k(i)} (rho_ij - k)`` over embedding neighbours
    that are not original neighbours, continuity likewise with ``r_ij`` over
    original neighbours lost in the embedding (Venna and Kaski 2001),
    ``Q_NX = (1/(nk)) sum_i |N_k(i) cap n_k(i)|`` (Lee and Verleysen 2009)
    and the local continuity meta-criterion ``LCMC = Q_NX - k/(n - 1)``
    (Chen and Buja 2009).

    References
    ----------
    Venna, J. and Kaski, S. (2001). Neighborhood preservation in nonlinear
    projection methods: an experimental study. *ICANN 2001*, LNCS 2130,
    485-491.
    Lee, J. A. and Verleysen, M. (2009). Quality assessment of dimensionality
    reduction: rank-based criteria. *Neurocomputing*, 72(7-9), 1431-1443.
    Chen, L. and Buja, A. (2009). Local multidimensional scaling for
    nonlinear dimension reduction, graph drawing, and proximity analysis.
    *Journal of the American Statistical Association*, 104(485), 209-219.

    Examples
    --------
    >>> D = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    >>> e = embedding_quality(D, D, 1)
    >>> e.trustworthiness, e.continuity, e.qnx
    (1.0, 1.0, 1.0)
    """
    Dh, Dl = _sq(D_high), _sq(D_low)
    n = len(Dh)
    if not 1 <= k < n / 2:
        raise ValueError("k must satisfy 1 <= k < n/2")
    Rh, Rl = _ranks(Dh), _ranks(Dl)
    t = c = 0.0
    q = 0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            inh, inl = Rh[i][j] <= k, Rl[i][j] <= k
            if inl and not inh:
                t += Rh[i][j] - k
            if inh and not inl:
                c += Rl[i][j] - k
            if inh and inl:
                q += 1
    norm = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    qnx = q / (n * k)
    return RichResult(
        payload={
            "trustworthiness": 1.0 - norm * t,
            "continuity": 1.0 - norm * c,
            "qnx": qnx,
            "lcmc": qnx - k / (n - 1.0),
        }
    )


def alienation_coefficient(dhat, d) -> float:
    r"""Guttman's coefficient of alienation ``K = sqrt(1 - mu^2)``, ``mu = sum d dhat / sqrt(sum d^2 sum dhat^2)``.

    ``mu`` is the congruence of the configuration distances with the
    disparities (Borg and Groenen 2005, section 11.1; Guttman 1968).

    References
    ----------
    Borg, I. and Groenen, P. J. F. (2005). *Modern Multidimensional Scaling:
    Theory and Applications*, 2nd edn. Springer, New York.

    Examples
    --------
    >>> round(alienation_coefficient([1.0, 2.0, 3.0], [1.1, 1.9, 3.2]), 6)
    0.052899
    """
    a = [float(v) for v in dhat]
    b = [float(v) for v in d]
    mu = ssum(x * y for x, y in zip(a, b)) / math.sqrt(ssum(x * x for x in a) * ssum(y * y for y in b))
    return math.sqrt(max(0.0, 1.0 - mu * mu))


def dissimilarity_check(delta, *, tol: float = 1e-12) -> RichResult:
    r"""Metric and Euclidean checks of a dissimilarity matrix.

    ``symmetric``, zero diagonal, non-negativity, the number of triangle
    inequality violations ``d_ij > d_ik + d_kj`` (with the largest excess)
    and Euclidean embeddability (Gower 1966: ``-J D^2 J / 2`` positive
    semi-definite), with the eigenvalues of that matrix.

    References
    ----------
    Gower, J. C. (1966). Some distance properties of latent root and vector
    methods used in multivariate analysis. *Biometrika*, 53(3-4), 325-338.

    Examples
    --------
    >>> c = dissimilarity_check([[0, 1, 5], [1, 0, 1], [5, 1, 0]])
    >>> c.triangle_violations, c.max_violation, c.euclidean
    (1, 3.0, False)
    """
    D = _sq(delta)
    n = len(D)
    viol, mx = 0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ex = max((D[i][j] - D[i][t] - D[t][j] for t in range(n) if t not in (i, j)), default=0.0)
            if ex > tol:
                viol += 1
                mx = max(mx, ex)
    D2 = [[v * v for v in r] for r in D]
    rm = [ssum(r) / n for r in D2]
    gm = ssum(rm) / n
    B = [[-0.5 * (D2[i][j] - rm[i] - rm[j] + gm) for j in range(n)] for i in range(n)]
    ev = sorted((float(v) for v in np.linalg.eigvalsh(np.asarray(B, dtype=float)).tolist()), reverse=True)
    scale = max(abs(v) for v in ev) or 1.0
    return RichResult(
        payload={
            "symmetric": all(D[i][j] == D[j][i] for i in range(n) for j in range(i)),
            "zero_diagonal": all(D[i][i] == 0 for i in range(n)),
            "nonnegative": all(v >= 0 for r in D for v in r),
            "triangle_violations": viol,
            "max_violation": mx,
            "euclidean": ev[-1] >= -1e-10 * scale,
            "eigenvalues": ev,
        }
    )


def cheatsheet() -> str:
    return "smacof / procrustes_fit / embedding_quality -> SMACOF MDS, Procrustes, trustworthiness and continuity."
