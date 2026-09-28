# morie.fn -- function file (rootcoder007/morie)
"""Wildlife ecology: minimum convex polygon home ranges, distance sampling (line and point transects,
half-normal and hazard-rate detection), single-season occupancy and N-mixture abundance models,
circuit-theory resistance and isolation by resistance, least-cost paths and corridors, habitat
suitability to resistance, habitat suitability indices, the partial Mantel test, gene flow from Fst
and Hanski's connectivity."""

from __future__ import annotations

import heapq
import math

from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._sci_core import _bfgs

__all__ = [
    "mcp_home_range",
    "distance_sampling",
    "occupancy_model",
    "nmixture_model",
    "circuit_resistance",
    "least_cost_path",
    "resistance_from_suitability",
    "habitat_suitability_index",
    "partial_mantel",
    "gene_flow_nm",
    "hanski_connectivity",
]


# ---------------------------------------------------------------- helpers


def _quantile7(v, q):
    s = sorted(v)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _hull(P):
    P = sorted(set(P))
    if len(P) <= 2:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _grad_hess(f, x, h=1e-4):
    k = len(x)
    g = [0.0] * k
    H = [[0.0] * k for _ in range(k)]
    f0 = f(x)
    for a in range(k):
        ha = h * max(1.0, abs(x[a]))
        xp, xm = list(x), list(x)
        xp[a] += ha
        xm[a] -= ha
        fp, fm = f(xp), f(xm)
        g[a] = (fp - fm) / (2 * ha)
        H[a][a] = (fp - 2 * f0 + fm) / ha**2
        for b in range(a):
            hb = h * max(1.0, abs(x[b]))
            v = []
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                t = list(x)
                t[a] += sa * ha
                t[b] += sb * hb
                v.append(f(t))
            H[a][b] = H[b][a] = (v[0] - v[1] - v[2] + v[3]) / (4 * ha * hb)
    return g, H


def _mle(nll0, x0):
    """Minimise ``nll``: BFGS, then Newton steps on central-difference derivatives until they stall."""

    def nll(t):
        try:
            v = nll0(t)
        except (OverflowError, ValueError, ZeroDivisionError):
            return math.inf
        return v if v == v else math.inf

    x = [float(v) for v in _bfgs(lambda t: nll(list(t)), list(x0), gtol=1e-10).x]
    f = nll(x)
    for _ in range(50):
        g, H = _grad_hess(nll, x)
        try:
            step = solve(H, g)
        except (ZeroDivisionError, ValueError):
            break
        xn = [a - b for a, b in zip(x, step)]
        fn = nll(xn)
        if not fn <= f:
            break
        done = max(abs(s) for s in step) < 1e-10
        x, f = xn, fn
        if done:
            break
    _, H = _grad_hess(nll, x)
    try:
        V = inverse(H)
        se = [math.sqrt(V[i][i]) if V[i][i] > 0 else math.nan for i in range(len(x))]
    except (ZeroDivisionError, ValueError):
        V, se = None, [math.nan] * len(x)
    return x, f, se, V


def _expit(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _log1pexp(z):
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def _gl(n=64):
    """Gauss-Legendre nodes/weights on [-1, 1] (Newton on P_n)."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


_GLX, _GLW = _gl(64)


def _integrate(f, a, b, pieces=16):
    tot = 0.0
    h = (b - a) / pieces
    for p in range(pieces):
        lo = a + p * h
        tot += ssum(w * f(lo + h * (x + 1) / 2) for x, w in zip(_GLX, _GLW)) * h / 2
    return tot


# ---------------------------------------------------------------- home range


def mcp_home_range(xy, percent: float = 95.0) -> RichResult:
    r"""Minimum convex polygon home range, as ``adehabitatHR::mcp``.

    Relocations farther from the arithmetic centroid than the ``percent``
    quantile (R type 7) of those distances are excluded; the home range is
    the convex hull of the rest (Mohr 1947; Calenge 2006). Area in squared
    coordinate units (and hectares for metres).

    References
    ----------
    Calenge, C. (2006). The package adehabitat for the R software: a tool for
    the analysis of space and habitat use by animals. *Ecological Modelling*,
    197(3-4), 516-519.

    Examples
    --------
    >>> r = mcp_home_range([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (10, 10)], percent=80)
    >>> r.area
    4.0
    """
    P = [(float(a), float(b)) for a, b in xy]
    if len(P) < 5:
        raise ValueError("at least 5 relocations are required")
    cx, cy = ssum(p[0] for p in P) / len(P), ssum(p[1] for p in P) / len(P)
    d = [math.hypot(p[0] - cx, p[1] - cy) for p in P]
    q = _quantile7(d, min(percent, 100.0) / 100)
    keep = [p for p, di in zip(P, d) if di <= q]
    H = _hull(keep)
    n = len(H)
    area = abs(ssum(H[i][0] * H[(i + 1) % n][1] - H[(i + 1) % n][0] * H[i][1] for i in range(n))) / 2
    return RichResult(payload={"area": area, "area_ha": area / 10000, "hull": H, "n_used": len(keep)})


# ---------------------------------------------------------------- distance sampling


def distance_sampling(
    distances,
    *,
    width: float,
    key: str = "hn",
    transect: str = "line",
    effort: float | None = None,
    area: float | None = None,
) -> RichResult:
    r"""Conventional distance sampling: detection function by maximum likelihood, detection probability, density.

    Key functions as ``mrds``: half-normal ``g(x) = exp(-x^2 / (2 sigma^2))``
    or hazard-rate ``g(x) = 1 - exp(-(x/sigma)^{-b})``, ``sigma = exp(beta_0)``,
    ``b = exp(beta_1)``, truncated at ``width``. Line transects use
    ``f(x) = g(x) / mu``, ``mu = int_0^w g`` (the effective strip half-width);
    point transects ``f(r) = 2 pi r g(r) / nu``, ``nu = 2 pi int_0^w r g(r)``.
    ``P = mu / w`` (line) or ``nu / (pi w^2)`` (point). With ``effort``
    (total line length, or number of points) the density is ``n / (2 L mu)`` or
    ``n / (k nu)``, times ``area`` for abundance (Buckland et al. 2001).

    References
    ----------
    Buckland, S. T., Anderson, D. R., Burnham, K. P., Laake, J. L.,
    Borchers, D. L. and Thomas, L. (2001). *Introduction to Distance
    Sampling*. Oxford University Press.

    Examples
    --------
    >>> r = distance_sampling([0.1, 0.4, 0.2, 0.8, 0.5, 0.05, 0.3, 1.1, 0.6, 0.25], width=1.5, effort=2.0)
    >>> round(r.p, 6), round(r.density, 4)
    (0.452987, 3.6793)
    """
    x = [float(v) for v in distances if float(v) <= width]
    n = len(x)
    if key not in ("hn", "hr") or transect not in ("line", "point"):
        raise ValueError("key must be hn or hr and transect line or point")

    def g(d, th):
        s = math.exp(th[0])
        if key == "hn":
            return math.exp(-d * d / (2 * s * s))
        if d <= 0:
            return 1.0
        t = -math.exp(th[1]) * math.log(d / s)  # log of (d/s)^(-b), kept finite
        return 1.0 if t > 700 else 1 - math.exp(-math.exp(t))

    def norm(th):
        if key == "hn":
            s = math.exp(th[0])
            if transect == "line":
                return s * math.sqrt(math.pi / 2) * math.erf(width / (s * math.sqrt(2)))
            return 2 * math.pi * s * s * (1 - math.exp(-width * width / (2 * s * s)))
        if transect == "line":
            return _integrate(lambda d: g(d, th), 0.0, width)
        return 2 * math.pi * _integrate(lambda d: d * g(d, th), 0.0, width)

    def nll(th):
        c = norm(th)
        if not c > 0:
            return math.inf
        ll = -n * math.log(c)
        for d in x:
            gd = g(d, th)
            if gd <= 0:
                return math.inf
            ll += math.log(gd) + (math.log(2 * math.pi * d) if transect == "point" and d > 0 else 0.0)
        return -ll

    s0 = math.sqrt(ssum(d * d for d in x) / n) if n else width / 2
    th0 = [math.log(s0)] + ([0.5] if key == "hr" else [])
    th, f, se, V = _mle(nll, th0)
    c = norm(th)
    P = c / width if transect == "line" else c / (math.pi * width * width)
    out = {
        "sigma": math.exp(th[0]),
        "shape": math.exp(th[1]) if key == "hr" else None,
        "coefficients": th,
        "se": se,
        "loglik": -f,
        "aic": 2 * f + 2 * len(th),
        "esw" if transect == "line" else "edr": c if transect == "line" else math.sqrt(c / math.pi),
        "p": P,
        "n": n,
    }
    if effort is not None:
        D = n / (2 * float(effort) * c) if transect == "line" else n / (float(effort) * c)
        out["density"] = D
        if area is not None:
            out["abundance"] = D * float(area)
    return RichResult(payload=out)


# ---------------------------------------------------------------- occupancy / N-mixture


def _design(X, n, name):
    if X is None:
        return [[1.0] for _ in range(n)]
    M = [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X]
    if len(M) != n:
        raise ValueError(f"{name} must have one row per site")
    return M


def occupancy_model(y, *, psi_covariates=None, p_covariates=None) -> RichResult:
    r"""Single-season site occupancy model (MacKenzie et al. 2002), as ``unmarked::occu``.

    ``y[i][j]`` is 1/0 detection at site ``i`` on visit ``j`` (``None`` for no
    visit). ``logit psi_i = X_i beta``, ``logit p_i = Z_i alpha`` (design
    rows default to an intercept); the site likelihood is ``psi_i prod_j
    p^y (1 - p)^{1 - y} + (1 - psi_i) 1[no detections]``. Maximum likelihood
    (BFGS then Newton), standard errors from the inverse Hessian; also the
    naive and model-based occupancy and each site's conditional occupancy
    probability.

    References
    ----------
    MacKenzie, D. I., Nichols, J. D., Lachman, G. B., Droege, S., Royle, J.
    A. and Langtimm, C. A. (2002). Estimating site occupancy rates when
    detection probabilities are less than one. *Ecology*, 83(8), 2248-2255.

    Examples
    --------
    >>> y = [[1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    >>> r = occupancy_model(y)
    >>> round(r.naive_occupancy, 6)
    0.5
    """
    Y = [[None if v is None or v != v else int(v) for v in row] for row in y]
    n = len(Y)
    Xs = _design(psi_covariates, n, "psi_covariates")
    Xd = _design(p_covariates, n, "p_covariates")
    q, r = len(Xs[0]), len(Xd[0])

    def nll(th):
        b, a = th[:q], th[q:]
        tot = 0.0
        for i in range(n):
            ps = _expit(ssum(u * v for u, v in zip(Xs[i], b)))
            eta = ssum(u * v for u, v in zip(Xd[i], a))
            obs = [v for v in Y[i] if v is not None]
            det = sum(obs)
            # log prod p^y (1-p)^(1-y), stable in eta
            lp = ssum(-_log1pexp(-eta) if v else -_log1pexp(eta) for v in obs)
            like = ps * math.exp(lp) + ((1 - ps) if det == 0 else 0.0)
            if like <= 0:
                return math.inf
            tot += math.log(like)
        return -tot

    th, f, se, V = _mle(nll, [0.0] * (q + r))
    b, a = th[:q], th[q:]
    psi = [_expit(ssum(u * v for u, v in zip(Xs[i], b))) for i in range(n)]
    p = [_expit(ssum(u * v for u, v in zip(Xd[i], a))) for i in range(n)]
    cond = []
    for i in range(n):
        obs = [v for v in Y[i] if v is not None]
        if sum(obs):
            cond.append(1.0)
        else:
            m = len(obs)
            num = psi[i] * (1 - p[i]) ** m
            cond.append(num / (num + 1 - psi[i]))
    naive = sum(1 for row in Y if any(v for v in row if v is not None)) / n
    return RichResult(
        payload={
            "psi_coefficients": b,
            "p_coefficients": a,
            "se": se,
            "loglik": -f,
            "aic": 2 * f + 2 * len(th),
            "psi": psi,
            "p": p,
            "conditional_occupancy": cond,
            "naive_occupancy": naive,
            "occupancy": ssum(psi) / n,
        }
    )


def nmixture_model(y, *, lambda_covariates=None, p_covariates=None, K: int | None = None) -> RichResult:
    r"""Binomial N-mixture abundance model (Royle 2004), Poisson mixture, as ``unmarked::pcount``.

    ``y[i][j]`` counts at site ``i`` on visit ``j``; ``N_i ~ Poisson(lambda_i)``,
    ``log lambda_i = X_i beta``, ``y_ij | N_i ~ Binomial(N_i, p_i)``, ``logit
    p_i = Z_i alpha``; the site likelihood sums ``N`` from ``max_j y_ij`` to
    ``K`` (default ``max(y) + 100``). Maximum likelihood with inverse-Hessian
    standard errors; returns the fitted ``lambda``, ``p`` and the total
    expected abundance.

    References
    ----------
    Royle, J. A. (2004). N-mixture models for estimating population size from
    spatially replicated counts. *Biometrics*, 60(1), 108-115.

    Examples
    --------
    >>> r = nmixture_model([[3, 2, 4], [0, 1, 0], [5, 3, 4], [2, 2, 1]])
    >>> r.K
    105
    """
    Y = [[None if v is None or v != v else int(v) for v in row] for row in y]
    n = len(Y)
    Xl = _design(lambda_covariates, n, "lambda_covariates")
    Xd = _design(p_covariates, n, "p_covariates")
    q, r = len(Xl[0]), len(Xd[0])
    K = max(v for row in Y for v in row if v is not None) + 100 if K is None else int(K)
    lf = [math.lgamma(k + 1) for k in range(K + 1)]

    def nll(th):
        b, a = th[:q], th[q:]
        tot = 0.0
        for i in range(n):
            ll = ssum(u * v for u, v in zip(Xl[i], b))
            lam = math.exp(ll)
            eta = ssum(u * v for u, v in zip(Xd[i], a))
            lp, lq = -_log1pexp(-eta), -_log1pexp(eta)
            obs = [v for v in Y[i] if v is not None]
            terms = []
            for N in range(max(obs) if obs else 0, K + 1):
                t = N * ll - lam - lf[N]
                for v in obs:
                    t += lf[N] - lf[v] - lf[N - v] + v * lp + (N - v) * lq
                terms.append(t)
            m = max(terms)
            tot += m + math.log(ssum(math.exp(t - m) for t in terms))
        return -tot

    th, f, se, V = _mle(nll, [0.0] * (q + r))
    b, a = th[:q], th[q:]
    lam = [math.exp(ssum(u * v for u, v in zip(Xl[i], b))) for i in range(n)]
    p = [_expit(ssum(u * v for u, v in zip(Xd[i], a))) for i in range(n)]
    return RichResult(
        payload={
            "lambda_coefficients": b,
            "p_coefficients": a,
            "se": se,
            "loglik": -f,
            "aic": 2 * f + 2 * len(th),
            "lambda": lam,
            "p": p,
            "total_abundance": ssum(lam),
            "K": K,
        }
    )


# ---------------------------------------------------------------- connectivity

_NB4 = ((0, 1), (1, 0))
_NB8 = ((0, 1), (1, 0), (1, 1), (1, -1))


def _grid(R):
    G = [[float(v) for v in row] for row in (R.tolist() if hasattr(R, "tolist") else R)]
    return G, len(G), len(G[0])


def circuit_resistance(resistance, nodes, *, directions: int = 4) -> RichResult:
    r"""Effective (circuit) resistance between cells of a resistance grid -- isolation by resistance (McRae 2006).

    Adjacent cells are joined by conductance ``1 / (d (r_a + r_b)/2)`` (``d`` =
    1, or ``sqrt 2`` for diagonals with ``directions=8``; Circuitscape's
    average-resistance rule; non-finite or ``None`` cells are barriers). The
    effective resistance between cells ``a`` and ``b`` is ``(e_a - e_b)' L^+
    (e_a - e_b)`` with ``L`` the graph Laplacian, solved with ``b`` grounded.
    ``nodes`` are (row, col) pairs; returns the pairwise matrix.

    References
    ----------
    McRae, B. H. (2006). Isolation by resistance. *Evolution*, 60(8),
    1551-1561.

    Examples
    --------
    >>> circuit_resistance([[1, 1, 1]], [(0, 0), (0, 2)]).resistance
    [[0.0, 2.0], [2.0, 0.0]]
    """
    G, nr, nc = _grid(resistance)
    ok = [[v == v and v is not None and math.isfinite(v) and v > 0 for v in row] for row in G]
    idx = {}
    for i in range(nr):
        for j in range(nc):
            if ok[i][j]:
                idx[(i, j)] = len(idx)
    m = len(idx)
    nb = _NB8 if directions == 8 else _NB4
    L = [[0.0] * m for _ in range(m)]
    for (i, j), u in idx.items():
        for di, dj in nb:
            x, y = i + di, j + dj
            if (x, y) in idx:
                v = idx[(x, y)]
                w = 1 / ((math.sqrt(2) if di and dj else 1.0) * (G[i][j] + G[x][y]) / 2)
                L[u][u] += w
                L[v][v] += w
                L[u][v] -= w
                L[v][u] -= w
    Nd = [idx[(int(a), int(b))] for a, b in nodes]
    k = len(Nd)
    R = [[0.0] * k for _ in range(k)]
    for s in range(k):
        for t in range(s + 1, k):
            gnd = Nd[t]
            keep = [u for u in range(m) if u != gnd]
            Lr = [[L[a][b] for b in keep] for a in keep]
            e = [1.0 if u == Nd[s] else 0.0 for u in keep]
            xv = solve(Lr, e)
            R[s][t] = R[t][s] = xv[keep.index(Nd[s])]
    return RichResult(payload={"resistance": R, "nodes": [tuple(n) for n in nodes]})


def _dijkstra(G, nr, nc, src, directions):
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (directions == 8 or not (a and b))]
    dist = [[math.inf] * nc for _ in range(nr)]
    prev = [[None] * nc for _ in range(nr)]
    si, sj = src
    dist[si][sj] = 0.0
    pq = [(0.0, si, sj)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i][j]:
            continue
        for di, dj in nb:
            x, y = i + di, j + dj
            if 0 <= x < nr and 0 <= y < nc and G[x][y] == G[x][y] and math.isfinite(G[x][y]):
                nd = d + (math.sqrt(2) if di and dj else 1.0) * (G[i][j] + G[x][y]) / 2
                if nd < dist[x][y] - 1e-15:
                    dist[x][y] = nd
                    prev[x][y] = (i, j)
                    heapq.heappush(pq, (nd, x, y))
    return dist, prev


def least_cost_path(cost, start, end, *, directions: int = 8, corridor_slack: float | None = None) -> RichResult:
    r"""Least-cost path between two cells of a cost (resistance) grid, and the least-cost corridor.

    Moving between adjacent cells costs ``d (c_a + c_b) / 2`` (``d`` = 1 or
    ``sqrt 2``), the transition used by ``gdistance`` and the terra cost
    distance; Dijkstra's algorithm gives the accumulated cost surfaces from
    both ends and the path. The corridor (Adriaensen et al. 2003) is the set
    of cells whose summed cost ``C_a + C_b`` is within ``corridor_slack`` of
    the least cost (default 10%).

    References
    ----------
    Adriaensen, F. et al. (2003). The application of 'least-cost' modelling
    as a functional landscape model. *Landscape and Urban Planning*, 64(4),
    233-247.

    Examples
    --------
    >>> r = least_cost_path([[1, 1, 1], [9, 9, 1], [1, 1, 1]], (0, 0), (2, 0), directions=4)
    >>> r.path, r.cost
    ([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)], 6.0)
    """
    G, nr, nc = _grid(cost)
    da, pa = _dijkstra(G, nr, nc, tuple(start), directions)
    db, _ = _dijkstra(G, nr, nc, tuple(end), directions)
    ei, ej = end
    path, cur = [], (ei, ej)
    while cur is not None:
        path.append(cur)
        cur = pa[cur[0]][cur[1]]
    path.reverse()
    best = da[ei][ej]
    slack = 0.1 * best if corridor_slack is None else float(corridor_slack)
    corr = [[da[i][j] + db[i][j] <= best + slack + 1e-12 for j in range(nc)] for i in range(nr)]
    return RichResult(
        payload={"path": path, "cost": best, "cost_from_start": da, "cost_from_end": db, "corridor": corr}
    )


def resistance_from_suitability(suitability, c: float = 8.0) -> list:
    r"""Landscape resistance from habitat suitability ``h`` in [0, 1] (Keeley, Beier and Gagnon 2016).

    ``R = 100 - 99 (1 - exp(-c h)) / (1 - exp(-c))``; ``c = 0`` is the linear
    transform ``100 - 99 h``, larger ``c`` makes resistance rise only in poor
    habitat.

    References
    ----------
    Keeley, A. T. H., Beier, P. and Gagnon, J. W. (2016). Estimating
    landscape resistance from habitat suitability: effects of data source and
    nonlinearities. *Landscape Ecology*, 31(9), 2151-2162.

    Examples
    --------
    >>> [round(v, 6) for v in resistance_from_suitability([0.0, 0.5, 1.0], c=0)]
    [100.0, 50.5, 1.0]
    """
    h = [float(v) for v in (suitability.tolist() if hasattr(suitability, "tolist") else suitability)]
    if c == 0:
        return [100 - 99 * v for v in h]
    return [100 - 99 * (1 - math.exp(-c * v)) / (1 - math.exp(-c)) for v in h]


def habitat_suitability_index(indices, *, method: str = "geometric", weights=None) -> list:
    r"""Habitat suitability index from suitability index variables in [0, 1] (U.S. Fish and Wildlife Service 1981).

    ``indices[k]`` holds variable ``k`` for every site. ``geometric`` =
    ``(prod SI_k^{w_k})^{1 / sum w}``, ``arithmetic`` the weighted mean,
    ``minimum`` the limiting factor.

    References
    ----------
    U.S. Fish and Wildlife Service (1981). *Standards for the Development of
    Habitat Suitability Index Models*. 103 ESM, Washington DC.

    Examples
    --------
    >>> [round(v, 6) for v in habitat_suitability_index([[0.5, 1.0], [0.8, 1.0]])]
    [0.632456, 1.0]
    """
    S = [[float(v) for v in col] for col in indices]
    w = [1.0] * len(S) if weights is None else [float(v) for v in weights]
    tw = ssum(w)
    out = []
    for i in range(len(S[0])):
        v = [col[i] for col in S]
        if method == "geometric":
            out.append(0.0 if min(v) <= 0 else math.exp(ssum(wk * math.log(x) for wk, x in zip(w, v)) / tw))
        elif method == "arithmetic":
            out.append(ssum(wk * x for wk, x in zip(w, v)) / tw)
        elif method == "minimum":
            out.append(min(v))
        else:
            raise ValueError("method must be geometric, arithmetic or minimum")
    return out


def _lower(D):
    n = len(D)
    return [float(D[i][j]) for j in range(n) for i in range(j + 1, n)]


def _pearson(a, b):
    ma, mb = ssum(a) / len(a), ssum(b) / len(b)
    sab = ssum((x - ma) * (y - mb) for x, y in zip(a, b))
    return sab / math.sqrt(ssum((x - ma) ** 2 for x in a) * ssum((y - mb) ** 2 for y in b))


def partial_mantel(A, B, C, *, nsim: int = 999, seed: int = 1) -> RichResult:
    r"""Partial Mantel test of distance matrices ``A`` and ``B`` controlling for ``C`` (isolation by environment).

    ``r(A,B|C) = (r_AB - r_AC r_BC) / sqrt((1 - r_AC^2)(1 - r_BC^2))`` over the
    lower triangles (Smouse, Long and Sokal 1986; ``vegan::mantel.partial``);
    the one-sided p-value permutes rows and columns of ``A`` together (Philox
    permutations).

    References
    ----------
    Smouse, P. E., Long, J. C. and Sokal, R. R. (1986). Multiple regression
    and correlation extensions of the Mantel test of matrix correspondence.
    *Systematic Zoology*, 35(4), 627-632.

    Examples
    --------
    >>> A = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    >>> C = [[0, 2, 1, 5], [2, 0, 3, 1], [1, 3, 0, 2], [5, 1, 2, 0]]
    >>> round(partial_mantel(A, A, C, nsim=0).statistic, 12)
    1.0
    """
    n = len(A)
    b, c = _lower(B), _lower(C)
    rbc = _pearson(b, c)

    def stat(Am):
        a = _lower(Am)
        rab, rac = _pearson(a, b), _pearson(a, c)
        return (rab - rac * rbc) / math.sqrt((1 - rac**2) * (1 - rbc**2))

    t = stat(A)
    sims = []
    for s in range(1, nsim + 1):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        p = list(range(n))
        for i in range(n - 1):
            k = i + int(u[i] * (n - i))
            p[i], p[k] = p[k], p[i]
        sims.append(stat([[A[p[i]][p[j]] for j in range(n)] for i in range(n)]))
    return RichResult(
        payload={"statistic": t, "pvalue": (1 + sum(1 for v in sims if v >= t)) / (nsim + 1) if nsim else math.nan}
    )


def gene_flow_nm(fst) -> float:
    r"""Number of migrants per generation from Fst under Wright's island model: ``Nm = (1/Fst - 1)/4``.

    References
    ----------
    Wright, S. (1931). Evolution in Mendelian populations. *Genetics*, 16(2),
    97-159.

    Examples
    --------
    >>> gene_flow_nm(0.2)
    1.0
    """
    return (1 / float(fst) - 1) / 4


def hanski_connectivity(coords, occupied, areas, *, alpha: float = 1.0, b: float = 0.5) -> list:
    r"""Hanski's (1994) patch connectivity ``S_i = sum_{j != i} p_j exp(-alpha d_ij) A_j^b``.

    ``occupied`` is 1/0 (or an occurrence probability), ``areas`` patch
    areas, ``alpha`` the inverse mean dispersal distance and ``b`` the
    emigration-area scaling.

    References
    ----------
    Hanski, I. (1994). A practical model of metapopulation dynamics. *Journal
    of Animal Ecology*, 63(1), 151-162.

    Examples
    --------
    >>> [round(v, 6) for v in hanski_connectivity([(0, 0), (1, 0)], [1, 1], [4, 9], alpha=1, b=0.5)]
    [1.103638, 0.735759]
    """
    P = [(float(a), float(bb)) for a, bb in coords]
    return [
        ssum(
            float(occupied[j]) * math.exp(-alpha * math.dist(P[i], P[j])) * float(areas[j]) ** b
            for j in range(len(P))
            if j != i
        )
        for i in range(len(P))
    ]


def cheatsheet() -> str:
    return (
        "mcp_home_range / distance_sampling / occupancy_model / nmixture_model / circuit_resistance / "
        "least_cost_path / resistance_from_suitability / habitat_suitability_index / partial_mantel / "
        "gene_flow_nm / hanski_connectivity -> wildlife ecology."
    )
