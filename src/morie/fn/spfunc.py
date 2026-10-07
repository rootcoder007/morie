# morie.fn -- function file (rootcoder007/morie)
"""Spatial functional, topological, graph and multilevel methods: functional PCA of curves,
graph convolution (GCN, SGC) and graph attention layers, Vietoris-Rips persistent homology and
persistence landscapes, possibilistic c-means, and REML variance components for crossed and
nested random effects."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult

__all__ = [
    "curve_fpca",
    "gcn_layer",
    "sgc_ridge",
    "gat_layer",
    "rips_persistence",
    "persistence_landscape",
    "possibilistic_cmeans",
    "reml_components",
    "crossed_random_effects",
    "nested_random_effects",
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


def curve_fpca(curves, t, n_components: int = 2) -> RichResult:
    r"""Functional principal components of curves observed on a common (possibly uneven) grid.

    With trapezoid quadrature weights ``w`` on ``t``, the covariance operator
    ``C(s, t) = (1/(N-1)) sum_i (x_i(s) - m(s)) (x_i(t) - m(t))`` is
    discretised as ``W^(1/2) C W^(1/2)``; its eigenvectors ``v`` give the
    eigenfunctions ``phi = W^(-1/2) v`` with ``int phi^2 = 1`` and the scores
    ``xi_ik = int (x_i - m) phi_k`` (Ramsay and Silverman 2005, ch. 8).
    Eigenfunction signs make the largest value positive.

    References
    ----------
    Ramsay, J. O. and Silverman, B. W. (2005). Functional Data Analysis, 2nd
    ed., ch. 8. Delicado, P., Giraldo, R., Comas, C. and Mateu, J. (2010).
    Statistics for spatial functional data. Environmetrics 21, 224-239.

    Examples
    --------
    >>> t = [0.0, 0.5, 1.0]
    >>> r = curve_fpca([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 1.0, 2.0]], t, 1)
    >>> [round(v, 12) for v in r.scores[0]]
    [0.0]
    """
    X = [[float(v) for v in row] for row in curves]
    N, M = len(X), len(t)
    w = [0.0] * M
    for j in range(M - 1):
        h = t[j + 1] - t[j]
        w[j] += h / 2
        w[j + 1] += h / 2
    m = [ssum(X[i][j] for i in range(N)) / N for j in range(M)]
    Z = [[X[i][j] - m[j] for j in range(M)] for i in range(N)]
    sw = [math.sqrt(v) for v in w]
    C = [[ssum(Z[i][a] * Z[i][b] for i in range(N)) / (N - 1) * sw[a] * sw[b] for b in range(M)] for a in range(M)]
    vals, vecs = _eig_desc(C)
    phis = [[vecs[k][j] / sw[j] for j in range(M)] for k in range(n_components)]
    scores = [[ssum(w[j] * Z[i][j] * phis[k][j] for j in range(M)) for k in range(n_components)] for i in range(N)]
    tot = ssum(v for v in vals if v > 0)
    return RichResult(
        payload={
            "mean": m,
            "eigenvalues": vals[:n_components],
            "eigenfunctions": phis,
            "scores": scores,
            "explained": [v / tot for v in vals[:n_components]],
        }
    )


def _norm_adj(A, self_loops):
    n = len(A)
    At = [[float(A[i][j]) + (1.0 if (self_loops and i == j) else 0.0) for j in range(n)] for i in range(n)]
    d = [ssum(r) for r in At]
    return [[At[i][j] / math.sqrt(d[i] * d[j]) if d[i] > 0 and d[j] > 0 else 0.0 for j in range(n)] for i in range(n)]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def _act(v, activation):
    if activation == "relu":
        return max(v, 0.0)
    if activation == "tanh":
        return math.tanh(v)
    return v


def gcn_layer(A, H, Wt, *, activation: str = "relu", self_loops: bool = True) -> list:
    r"""Graph convolution layer ``sigma(D~^(-1/2) A~ D~^(-1/2) H W)`` (Kipf and Welling 2017).

    ``A~ = A + I`` (with ``self_loops``) and ``D~`` its degree matrix;
    ``H`` holds node features (rows are nodes), ``Wt`` the layer weights;
    ``activation`` is "relu", "tanh" or "linear".

    References
    ----------
    Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with
    graph convolutional networks. ICLR 2017.

    Examples
    --------
    >>> gcn_layer([[0, 1], [1, 0]], [[1.0], [3.0]], [[1.0]])
    [[2.0], [2.0]]
    """
    S = _norm_adj(A, self_loops)
    Z = _mm(_mm(S, [[float(v) for v in r] for r in H]), [[float(v) for v in r] for r in Wt])
    return [[_act(v, activation) for v in row] for row in Z]


def sgc_ridge(A, X, y, *, k: int = 2, l2: float = 1.0, self_loops: bool = True) -> RichResult:
    r"""Simplified graph convolution (SGC) regression: ridge regression on ``S^K X``.

    ``S = D~^(-1/2) A~ D~^(-1/2)``; the smoothed features ``S^K X`` (with an
    intercept) are fitted by ridge regression ``(F'F + l2 I_0)^(-1) F'y``
    (intercept unpenalised) (Wu et al. 2019).

    References
    ----------
    Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T. and Weinberger, K. (2019).
    Simplifying graph convolutional networks. ICML 2019, 6861-6871.

    Examples
    --------
    >>> r = sgc_ridge([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], k=1, l2=0.0)
    >>> [round(v, 10) for v in r.fitted]
    [0.9917281742, 2.4912958027, 2.5169760231]
    """
    S = _norm_adj(A, self_loops)
    F = [[float(v) for v in r] for r in X]
    for _ in range(k):
        F = _mm(S, F)
    Fr = [[1.0] + r for r in F]
    p = len(Fr[0])
    G = [[ssum(r[a] * r[b] for r in Fr) + (l2 if (a == b and a > 0) else 0.0) for b in range(p)] for a in range(p)]
    beta = solve(G, [ssum(r[a] * v for r, v in zip(Fr, y)) for a in range(p)])
    fitted = [ssum(r[a] * beta[a] for a in range(p)) for r in Fr]
    return RichResult(payload={"coefficients": beta, "features": F, "fitted": fitted})


def gat_layer(
    A, H, Wt, a, *, negative_slope: float = 0.2, activation: str = "linear", self_loops: bool = True
) -> RichResult:
    r"""Graph attention layer (Velickovic et al. 2018), one head.

    ``z_i = W' h_i``; ``e_ij = LeakyReLU(a' [z_i || z_j])`` for neighbours
    ``j`` of ``i`` (and ``i`` itself with ``self_loops``);
    ``alpha_ij = softmax_j(e_ij)``; output ``sigma(sum_j alpha_ij z_j)``.

    References
    ----------
    Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and
    Bengio, Y. (2018). Graph attention networks. ICLR 2018.

    Examples
    --------
    >>> r = gat_layer([[0, 1], [1, 0]], [[1.0], [3.0]], [[1.0]], [0.0, 0.0])
    >>> r.output
    [[2.0], [2.0]]
    """
    Z = _mm([[float(v) for v in r] for r in H], [[float(v) for v in r] for r in Wt])
    n, f = len(Z), len(Z[0])
    att = [[0.0] * n for _ in range(n)]
    out = []
    for i in range(n):
        nb = [j for j in range(n) if (A[i][j] != 0 and j != i) or (self_loops and j == i)]
        e = []
        for j in nb:
            s = ssum(a[q] * Z[i][q] for q in range(f)) + ssum(a[f + q] * Z[j][q] for q in range(f))
            e.append(s if s > 0 else negative_slope * s)
        mx = max(e)
        ex = [math.exp(v - mx) for v in e]
        tot = ssum(ex)
        for j, v in zip(nb, ex):
            att[i][j] = v / tot
        out.append([_act(ssum(att[i][j] * Z[j][q] for j in nb), activation) for q in range(f)])
    return RichResult(payload={"output": out, "attention": att})


def _dist(points):
    n = len(points)
    return [[math.sqrt(ssum((a - b) ** 2 for a, b in zip(points[i], points[j]))) for j in range(n)] for i in range(n)]


def rips_persistence(
    points, *, max_dim: int = 1, max_scale: float = math.inf, distance_matrix: bool = False
) -> RichResult:
    r"""Persistent homology (dimensions 0 and 1) of the Vietoris-Rips filtration over Z/2.

    Simplices (vertices at 0, edges at their length, triangles at their
    longest edge, up to ``max_scale``) are ordered by value, then dimension,
    then vertex indices, and the boundary matrix is reduced by the standard
    column algorithm; each pivot ``low(j) = i`` pairs a birth at simplex
    ``i`` with a death at ``j``. Zero-persistence pairs are dropped;
    unpaired classes die at ``inf``. Returns rows ``(dimension, birth, death)``.

    References
    ----------
    Edelsbrunner, H., Letscher, D. and Zomorodian, A. (2002). Topological
    persistence and simplification. Discrete Comput. Geom. 28, 511-533.
    Zomorodian, A. and Carlsson, G. (2005). Computing persistent homology.
    Discrete Comput. Geom. 33, 249-274.

    Examples
    --------
    >>> sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    >>> d = rips_persistence(sq).diagram
    >>> [r for r in d if r[0] == 1]
    [[1, 1.0, 1.4142135623730951]]
    """
    D = [[float(v) for v in r] for r in points] if distance_matrix else _dist(points)
    n = len(D)
    simp = [(0.0, 0, (i,)) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] <= max_scale:
                simp.append((D[i][j], 1, (i, j)))
    if max_dim >= 1:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v = max(D[i][j], D[i][k], D[j][k])
                    if v <= max_scale:
                        simp.append((v, 2, (i, j, k)))
    simp.sort(key=lambda s: (s[0], s[1], s[2]))
    index = {s[2]: q for q, s in enumerate(simp)}
    cols = []
    for _val, dim, vs in simp:
        if dim == 0:
            cols.append([])
        else:
            faces = [tuple(v for v in vs if v != x) for x in vs]
            cols.append(sorted(index[f] for f in faces))
    pivot_of = {}
    pairs = []
    paired = set()
    for j in range(len(cols)):
        c = cols[j]
        while c and c[-1] in pivot_of:
            other = cols[pivot_of[c[-1]]]
            sa, sb = set(c), set(other)
            c = sorted(sa ^ sb)
        cols[j] = c
        if c:
            low = c[-1]
            pivot_of[low] = j
            paired.add(low)
            paired.add(j)
            b, dth = simp[low][0], simp[j][0]
            if dth > b:
                pairs.append([simp[low][1], b, dth])
    for q, s in enumerate(simp):
        if q not in paired and not cols[q] and s[1] <= max_dim:
            pairs.append([s[1], s[0], math.inf])
    pairs = [p for p in pairs if p[0] <= max_dim]
    pairs.sort(key=lambda r: (r[0], r[1], r[2]))
    return RichResult(payload={"diagram": pairs})


def persistence_landscape(diagram, t, *, k_max: int = 3, dimension: int | None = None) -> list:
    r"""Persistence landscape functions ``lambda_k(t)`` (Bubenik 2015).

    For finite pairs ``(b, d)`` the tent ``f(t) = max(0, min(t - b, d - t))``;
    ``lambda_k(t)`` is the ``k``-th largest tent value at ``t``. ``diagram``
    rows are ``(dimension, birth, death)`` (filtered by ``dimension``) or
    ``(birth, death)``. Returns ``k_max`` lists over the grid ``t``.

    References
    ----------
    Bubenik, P. (2015). Statistical topological data analysis using
    persistence landscapes. JMLR 16, 77-102.

    Examples
    --------
    >>> persistence_landscape([[0.0, 2.0], [1.0, 3.0]], [0.5, 1.0, 1.5, 2.0], k_max=2)
    [[0.5, 1.0, 0.5, 1.0], [0.0, 0.0, 0.5, 0.0]]
    """
    pairs = []
    for r in diagram:
        if len(r) == 3:
            if dimension is not None and r[0] != dimension:
                continue
            b, d = float(r[1]), float(r[2])
        else:
            b, d = float(r[0]), float(r[1])
        if math.isfinite(d):
            pairs.append((b, d))
    out = [[0.0] * len(t) for _ in range(k_max)]
    for q, tv in enumerate(t):
        vals = sorted((max(0.0, min(tv - b, d - tv)) for b, d in pairs), reverse=True)
        for k in range(min(k_max, len(vals))):
            out[k][q] = vals[k]
    return out


def _sqd(a, b):
    return ssum((u - v) ** 2 for u, v in zip(a, b))


def possibilistic_cmeans(
    X, centers, *, m: float = 2.0, omega=None, K: float = 1.0, max_iter: int = 1000, con_val: float = 1e-9
) -> RichResult:
    r"""Possibilistic c-means clustering (Krishnapuram and Keller 1993), as ``ppclust::pcm``.

    Typicalities ``t_ij = 1 / (1 + (d_ij / omega_j)^(1/(m-1)))`` with squared
    Euclidean ``d_ij``, prototypes ``v_j = sum_i t_ij^m x_i / sum_i t_ij^m``,
    iterated from ``centers`` until ``sum |v - v_old| <= con_val``. When
    ``omega`` is not given, fuzzy c-means (Bezdek) is run from ``centers``
    and ``omega_j = K sum_i u_ij^m d_ij / sum_i u_ij^m``, and PCM starts from
    the FCM prototypes.

    References
    ----------
    Krishnapuram, R. and Keller, J. M. (1993). A possibilistic approach to
    clustering. IEEE Trans. Fuzzy Systems 1, 98-110. Bezdek, J. C. (1981).
    Pattern Recognition with Fuzzy Objective Function Algorithms. Plenum.

    Examples
    --------
    >>> X = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
    >>> r = possibilistic_cmeans(X, [[0.0, 0.1], [5.0, 5.1]])
    >>> [round(v, 6) for v in r.centers[0]]
    [0.095509, 0.0]
    """
    Xs = [[float(v) for v in r] for r in X]
    n, k, p = len(Xs), len(centers), len(Xs[0])
    v = [[float(a) for a in r] for r in centers]
    if omega is None:
        # fuzzy c-means from the given prototypes
        prev = None
        for _ in range(max_iter):
            d = [[_sqd(Xs[i], v[j]) for j in range(k)] for i in range(n)]
            u = []
            for i in range(n):
                if any(dv == 0.0 for dv in d[i]):
                    u.append([1.0 if dv == 0.0 else 0.0 for dv in d[i]])
                else:
                    u.append([1.0 / ssum((d[i][j] / d[i][q]) ** (1.0 / (m - 1)) for q in range(k)) for j in range(k)])
            v = [
                [
                    ssum(u[i][j] ** m * Xs[i][c] for i in range(n)) / ssum(u[i][j] ** m for i in range(n))
                    for c in range(p)
                ]
                for j in range(k)
            ]
            if prev is not None and ssum(abs(a - b) for r1, r2 in zip(v, prev) for a, b in zip(r1, r2)) <= con_val:
                break
            prev = [list(r) for r in v]
        d = [[_sqd(Xs[i], v[j]) for j in range(k)] for i in range(n)]
        omega = [
            K * ssum(u[i][j] ** m * d[i][j] for i in range(n)) / ssum(u[i][j] ** m for i in range(n)) for j in range(k)
        ]
    omega = [float(o) for o in omega]
    it = 0
    change = math.inf
    t = [[0.0] * k for _ in range(n)]
    while change > con_val and it < max_iter:
        it += 1
        prev = [list(r) for r in v]
        d = [[_sqd(Xs[i], v[j]) for j in range(k)] for i in range(n)]
        t = [[1.0 / (1.0 + (d[i][j] / omega[j]) ** (1.0 / (m - 1))) for j in range(k)] for i in range(n)]
        v = [
            [ssum(t[i][j] ** m * Xs[i][c] for i in range(n)) / ssum(t[i][j] ** m for i in range(n)) for c in range(p)]
            for j in range(k)
        ]
        change = ssum(abs(a - b) for r1, r2 in zip(v, prev) for a, b in zip(r1, r2))
    return RichResult(payload={"centers": v, "typicality": t, "omega": omega, "iterations": it})


def reml_components(y, X, Zs, *, max_iter: int = 200, tol: float = 1e-12) -> RichResult:
    r"""REML variance components of ``y = X beta + sum_k Z_k u_k + e`` by average-information Newton steps.

    ``V = sum_k s_k Z_k Z_k' + s_e I``. The restricted log-likelihood
    ``-(1/2)(log|V| + log|X'V^(-1)X| + y'Py)`` has score
    ``-(1/2) tr(P V_k) + (1/2) y'P V_k P y`` and average information
    ``(1/2) y'P V_k P V_l P y`` (``P = V^(-1) - V^(-1)X(X'V^(-1)X)^(-1)X'V^(-1)``);
    steps are halved until the restricted likelihood does not decrease, and
    variances are kept at or above ``1e-10 var(y)`` (a component estimated
    on the boundary sits at that floor and leaves the Newton system while its
    score is negative). Returns the variances
    (``Z`` terms then residual), ``beta`` (GLS), BLUPs ``s_k Z_k'Py`` and the
    restricted log-likelihood (with its ``-(n - p)/2 log(2 pi)`` constant).

    References
    ----------
    Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block
    information when block sizes are unequal. Biometrika 58, 545-554.
    Gilmour, A. R., Thompson, R. and Cullis, B. R. (1995). Average information
    REML. Biometrics 51, 1440-1450.

    Examples
    --------
    >>> y = [1.0, 1.2, 3.0, 3.3, 2.0, 2.4]
    >>> Z = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    >>> r = reml_components(y, [[1.0]] * 6, [Z])
    >>> [round(v, 6) for v in r.variances]
    [1.028333, 0.048333]
    """
    ys = [float(v) for v in y]
    n = len(ys)
    Xm = [[float(v) for v in r] for r in X]
    p = len(Xm[0])
    ZZ = [
        [[ssum(float(Z[i][c]) * float(Z[j][c]) for c in range(len(Z[0]))) for j in range(n)] for i in range(n)]
        for Z in Zs
    ]
    ZZ.append([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
    K = len(ZZ)
    m = ssum(ys) / n
    s = [ssum((v - m) ** 2 for v in ys) / (n - 1) / K] * K

    def pieces(s):
        V = [[ssum(s[q] * ZZ[q][i][j] for q in range(K)) for j in range(n)] for i in range(n)]
        Vi = inverse(V)
        VX = _mm(Vi, Xm)
        XVX = [[ssum(Xm[r][a] * VX[r][b] for r in range(n)) for b in range(p)] for a in range(p)]
        XVXi = inverse(XVX)
        P = [
            [
                Vi[i][j] - ssum(VX[i][a] * ssum(XVXi[a][b] * VX[j][b] for b in range(p)) for a in range(p))
                for j in range(n)
            ]
            for i in range(n)
        ]
        Py = [ssum(P[i][j] * ys[j] for j in range(n)) for i in range(n)]
        return V, Vi, XVX, XVXi, P, Py, VX

    def rll(V, XVX, Py):
        _, ldv = _slogdet(V)
        _, ldx = _slogdet(XVX)
        return -0.5 * (ldv + ldx + ssum(a * b for a, b in zip(ys, Py))) - 0.5 * (n - p) * math.log(2 * math.pi)

    floor = 1e-10 * ssum((v - m) ** 2 for v in ys) / (n - 1)
    V, Vi, XVX, XVXi, P, Py, VX = pieces(s)
    ll = rll(V, XVX, Py)
    it = 0
    for _ in range(max_iter):
        it += 1
        VkPy = [[ssum(ZZ[q][i][j] * Py[j] for j in range(n)) for i in range(n)] for q in range(K)]
        PVkPy = [[ssum(P[i][j] * VkPy[q][j] for j in range(n)) for i in range(n)] for q in range(K)]
        score = [
            -0.5 * ssum(P[i][j] * ZZ[q][j][i] for i in range(n) for j in range(n))
            + 0.5 * ssum(a * b for a, b in zip(Py, VkPy[q]))
            for q in range(K)
        ]
        AI = [[0.5 * ssum(a * b for a, b in zip(VkPy[q], PVkPy[r])) for r in range(K)] for q in range(K)]
        # active set: components held at the floor with a negative score stay there
        free = [q for q in range(K) if s[q] > floor * (1 + 1e-9) or score[q] > 0]
        sf = solve([[AI[q][r] for r in free] for q in free], [score[q] for q in free]) if free else []
        step = [0.0] * K
        for q, v in zip(free, sf):
            step[q] = v
        f = 1.0
        accepted = None
        for _h in range(40):
            new = [max(a + f * b, floor) for a, b in zip(s, step)]
            pc = pieces(new)
            ll_new = rll(pc[0], pc[2], pc[5])
            if ll_new >= ll - 1e-12 * abs(ll):
                accepted = (new, pc, ll_new)
                break
            f /= 2.0
        if accepted is None:
            break
        new, pc, ll_new = accepted
        delta = max(abs(a - b) / max(a, floor) for a, b in zip(new, s))
        s = new
        V, Vi, XVX, XVXi, P, Py, VX = pc
        ll = ll_new
        if delta <= tol:
            break
    Vy = [ssum(Vi[i][j] * ys[j] for j in range(n)) for i in range(n)]
    beta = [ssum(XVXi[a][b] * ssum(Xm[r][b] * Vy[r] for r in range(n)) for b in range(p)) for a in range(p)]
    blups = [
        [s[q] * ssum(float(Zs[q][i][c]) * Py[i] for i in range(n)) for c in range(len(Zs[q][0]))] for q in range(K - 1)
    ]
    return RichResult(payload={"variances": s, "beta": beta, "blups": blups, "reml_loglik": ll, "iterations": it})


def _slogdet(M):
    A = [list(r) for r in M]
    n = len(A)
    s = 0.0
    sign = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if p != c:
            A[c], A[p] = A[p], A[c]
            sign = -sign
        piv = A[c][c]
        if piv < 0:
            sign = -sign
        s += math.log(abs(piv))
        for r in range(c + 1, n):
            f = A[r][c] / piv
            if f != 0.0:
                for q in range(c, n):
                    A[r][q] -= f * A[c][q]
    return sign, s


def _indicator(codes):
    levels = sorted(set(codes), key=lambda v: (str(type(v)), v))
    pos = {v: q for q, v in enumerate(levels)}
    return [[1.0 if pos[c] == q else 0.0 for q in range(len(levels))] for c in codes], levels


def crossed_random_effects(y, a, b, X=None) -> RichResult:
    r"""Crossed random effects ``y = X beta + u_a + v_b + e`` by REML (two-way crossed classification).

    ``a`` and ``b`` are factor codes; ``X`` defaults to an intercept. Calls
    :func:`reml_components` with the two indicator designs.

    Examples
    --------
    >>> y = [1.0, 2.1, 1.4, 2.6, 0.9, 2.2]
    >>> r = crossed_random_effects(y, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])
    >>> len(r.variances)
    3
    """
    Za, _ = _indicator(a)
    Zb, _ = _indicator(b)
    Xm = [[1.0] for _ in y] if X is None else X
    return reml_components(y, Xm, [Za, Zb])


def nested_random_effects(y, a, b, X=None) -> RichResult:
    r"""Nested random effects ``y = X beta + u_a + v_(b within a) + e`` by REML.

    Groups ``b`` are identified within ``a`` (the pair ``(a, b)``). Calls
    :func:`reml_components` with the two indicator designs.

    Examples
    --------
    >>> y = [1.0, 1.2, 1.9, 2.3, 4.0, 4.1, 3.2, 3.6]
    >>> r = nested_random_effects(y, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1])
    >>> len(r.variances)
    3
    """
    Za, _ = _indicator(a)
    Zb, _ = _indicator([(p, q) for p, q in zip(a, b)])
    Xm = [[1.0] for _ in y] if X is None else X
    return reml_components(y, Xm, [Za, Zb])


def cheatsheet() -> str:
    return (
        "curve_fpca / gcn_layer / sgc_ridge / gat_layer / rips_persistence / persistence_landscape / "
        "possibilistic_cmeans / reml_components / crossed_random_effects / nested_random_effects -> "
        "spatial functional, topological, graph and multilevel methods."
    )


# alias kept from the retired placeholder of the same name
fda_spatial = curve_fpca

# alias kept from the retired placeholder of the same name
fpca_spatial = curve_fpca

# alias kept from the retired placeholder of the same name
graph_attention_sp = gat_layer

# alias kept from the retired placeholder of the same name
graph_conv_sp = gcn_layer

# alias kept from the retired placeholder of the same name
hier_spatial_cross = crossed_random_effects

# alias kept from the retired placeholder of the same name
hier_spatial_fe = nested_random_effects

# alias kept from the retired placeholder of the same name
persistence_land = persistence_landscape

# alias kept from the retired placeholder of the same name
possibilistic_sp = possibilistic_cmeans

# alias kept from the retired placeholder of the same name
tda_persistent = rips_persistence
