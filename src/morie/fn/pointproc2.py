# morie.fn -- function file (rootcoder007/morie)
"""Point processes: area-interaction simulation by birth-death Metropolis-Hastings, Thomas
cluster (Cox) processes, log-Gaussian Cox processes on a grid, Abramson adaptive-kernel intensity,
penalised Berman-Turner log-linear intensity fits, and space-time K/L, G, F and J functions."""

from __future__ import annotations

import math

from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform

__all__ = [
    "area_interaction_simulate",
    "lgcp_simulate_grid",
    "abramson_intensity",
    "berman_turner_fit",
    "st_k_function",
    "st_g_function",
    "st_j_function",
]


def _poisson(lam, u):
    # inversion of the Poisson CDF at a uniform
    k, p = 0, math.exp(-lam)
    c = p
    while u > c and k < 100000:
        k += 1
        p *= lam / k
        c += p
    return k


def area_interaction_simulate(
    beta: float, eta: float, r: float, window, n_steps: int, *, grid: int = 100, seed: int = 0
) -> RichResult:
    r"""Area-interaction process (Baddeley and van Lieshout 1995) by birth-death Metropolis-Hastings.

    Density ``f(x) ~ beta^n(x) eta^(-C(x))`` with ``C(x)`` the area of the
    union of discs of radius ``r`` about the points divided by ``pi r^2``
    (union area counted on a ``grid x grid`` lattice over the dilated window);
    ``eta > 1`` attracts, ``eta < 1`` inhibits. Each step (Philox stream
    ``s``, four uniforms) proposes a birth or a death with probability 1/2,
    accepted with ``beta |W| eta^(-dC) / (n + 1)`` or its reciprocal (Geyer
    and Moller 1994).

    References
    ----------
    Baddeley, A. J. and van Lieshout, M. N. M. (1995). Area-interaction point
    processes. Ann. Inst. Statist. Math. 47, 601-619. Geyer, C. J. and
    Moller, J. (1994). Simulation procedures and likelihood inference for
    spatial point processes. Scand. J. Statist. 21, 359-373.

    Examples
    --------
    >>> r = area_interaction_simulate(50.0, 1.0, 0.05, (0, 1, 0, 1), 200, grid=20, seed=1)
    >>> r.n > 0
    True
    """
    x0, x1, y0, y1 = window
    area = (x1 - x0) * (y1 - y0)
    gx0, gy0 = x0 - r, y0 - r
    dx, dy = (x1 - x0 + 2 * r) / grid, (y1 - y0 + 2 * r) / grid
    cover = [[0] * grid for _ in range(grid)]
    unit = dx * dy / (math.pi * r * r)

    def cells(px, py):
        out = []
        i0, i1 = max(0, int((px - r - gx0) / dx)), min(grid - 1, int((px + r - gx0) / dx))
        j0, j1 = max(0, int((py - r - gy0) / dy)), min(grid - 1, int((py + r - gy0) / dy))
        for i in range(i0, i1 + 1):
            cx = gx0 + (i + 0.5) * dx
            for j in range(j0, j1 + 1):
                cy = gy0 + (j + 0.5) * dy
                if (cx - px) ** 2 + (cy - py) ** 2 <= r * r:
                    out.append((i, j))
        return out

    pts = []
    for s in range(n_steps):
        u = random_uniform(4, seed=seed, stream=s)
        if float(u[0]) < 0.5:
            px, py = x0 + float(u[1]) * (x1 - x0), y0 + float(u[2]) * (y1 - y0)
            cl = cells(px, py)
            dC = sum(1 for i, j in cl if cover[i][j] == 0) * unit
            ratio = beta * area * eta ** (-dC) / (len(pts) + 1)
            if float(u[3]) < ratio:
                pts.append((px, py))
                for i, j in cl:
                    cover[i][j] += 1
        elif pts:
            k = min(int(float(u[1]) * len(pts)), len(pts) - 1)
            cl = cells(*pts[k])
            dC = sum(1 for i, j in cl if cover[i][j] == 1) * unit
            ratio = len(pts) / (beta * area) * eta**dC
            if float(u[3]) < ratio:
                for i, j in cl:
                    cover[i][j] -= 1
                pts.pop(k)
    covered = sum(1 for row in cover for c in row if c > 0) * unit
    return RichResult(payload={"points": [list(p) for p in pts], "n": len(pts), "union_area_units": covered})


def lgcp_simulate_grid(
    nx: int, ny: int, window, mu: float, sigma2: float, scale: float, *, seed: int = 0
) -> RichResult:
    r"""Log-Gaussian Cox process on a grid (Moller, Syversveen and Waagepetersen 1998).

    ``Z`` is a Gaussian field on the cell centres with exponential covariance
    ``sigma2 exp(-d / scale)`` (Cholesky factor times Philox normals, stream
    0); cell counts are ``Poisson(exp(mu + Z) |cell|)`` (uniforms on stream
    1). The pair correlation is ``g(r) = exp(sigma2 exp(-r / scale))``.

    References
    ----------
    Moller, J., Syversveen, A. R. and Waagepetersen, R. P. (1998). Log
    Gaussian Cox processes. Scand. J. Statist. 25, 451-482.

    Examples
    --------
    >>> r = lgcp_simulate_grid(3, 2, (0, 3, 0, 2), 1.0, 0.5, 1.0, seed=3)
    >>> len(r.counts), len(r.intensity)
    (6, 6)
    """
    x0, x1, y0, y1 = window
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    cen = [(x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy) for j in range(ny) for i in range(nx)]
    n = len(cen)
    C = [[sigma2 * math.exp(-math.hypot(a[0] - b[0], a[1] - b[1]) / scale) for b in cen] for a in cen]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    z = [float(v) for v in random_normal(n, seed=seed, stream=0)]
    Z = [ssum(L[i][k] * z[k] for k in range(i + 1)) for i in range(n)]
    lam = [math.exp(mu + v) for v in Z]
    u = random_uniform(n, seed=seed, stream=1)
    counts = [_poisson(lam[i] * dx * dy, float(u[i])) for i in range(n)]
    return RichResult(payload={"field": Z, "intensity": lam, "counts": counts, "centres": [list(c) for c in cen]})


def _gk(d2, h):
    return math.exp(-d2 / (2 * h * h)) / (2 * math.pi * h * h)


def abramson_intensity(points, at, h0: float, *, trim: float = 5.0) -> RichResult:
    r"""Adaptive-kernel intensity with Abramson's square-root law (no edge correction).

    A fixed-bandwidth Gaussian pilot ``f(x_i) = sum_j K_h0(x_i - x_j)``
    gives local bandwidths ``h_i = h0 min((f(x_i) / G)^(-1/2), trim)`` with
    ``G`` the geometric mean of the pilot values; the intensity is
    ``lambda(u) = sum_i K_(h_i)(u - x_i)`` (as ``spatstat``'s ``bw.abram``
    with ``densityAdaptiveKernel`` and ``edge = FALSE``).

    References
    ----------
    Abramson, I. S. (1982). On bandwidth variation in kernel estimates - a
    square root law. Ann. Statist. 10, 1217-1223. Davies, T. M. and
    Baddeley, A. (2018). Fast computation of spatially adaptive kernel
    estimates. Statistics and Computing 28, 937-956.

    Examples
    --------
    >>> r = abramson_intensity([(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0)], 0.5)
    >>> [round(v, 6) for v in r.bandwidths]
    [0.5, 0.5]
    """
    P = [(float(a), float(b)) for a, b in points]
    n = len(P)
    pilot = [ssum(_gk((P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2, h0) for j in range(n)) for i in range(n)]
    G = math.exp(ssum(math.log(v) for v in pilot) / n)
    h = [h0 * min((v / G) ** -0.5, trim) for v in pilot]
    lam = [ssum(_gk((u[0] - P[i][0]) ** 2 + (u[1] - P[i][1]) ** 2, h[i]) for i in range(n)) for u in at]
    return RichResult(payload={"intensity": lam, "bandwidths": h, "pilot": pilot})


def berman_turner_fit(
    points, window, covariates, nx: int, ny: int, *, ridge: float = 0.0, max_iter: int = 100
) -> RichResult:
    r"""Log-linear Poisson intensity ``lambda(u) = exp(theta' z(u))`` by penalised Berman-Turner quadrature.

    Quadrature points are the data plus the ``nx x ny`` tile centres; each
    tile's area is shared equally among the quadrature points in it (the
    "grid" counting weights of Baddeley and Turner 2000; tiles are
    right-closed as in spatstat). With ``y_j = z_j / w_j``
    the weighted Poisson log-likelihood ``sum_j w_j (y_j eta_j - exp(eta_j))
    - (ridge / 2) ||theta_(-1)||^2`` is maximised by Newton-Raphson (IRLS).
    ``covariates(x, y)`` returns the covariate vector (an intercept is added).

    References
    ----------
    Berman, M. and Turner, T. R. (1992). Approximating point process
    likelihoods with GLIM. Applied Statistics 41, 31-38. Baddeley, A. and
    Turner, R. (2000). Practical maximum pseudolikelihood for spatial point
    patterns. Aust. N. Z. J. Statist. 42, 283-322.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.7, 0.5), (0.9, 0.9)]
    >>> r = berman_turner_fit(pts, (0, 1, 0, 1), lambda x, y: [], 2, 2)
    >>> round(math.exp(r.coefficients[0]), 10)
    4.0
    """
    x0, x1, y0, y1 = window
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    data = [(float(a), float(b)) for a, b in points]
    dummy = [(x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy) for j in range(ny) for i in range(nx)]
    Q = data + dummy
    z = [1.0] * len(data) + [0.0] * len(dummy)

    def tile(p):
        # right-closed tiles, as spatstat's grid1index: ceiling(n (x - x0) / width), clamped
        i = min(max(math.ceil(nx * (p[0] - x0) / (x1 - x0)), 1), nx) - 1
        j = min(max(math.ceil(ny * (p[1] - y0) / (y1 - y0)), 1), ny) - 1
        return i + nx * j

    tiles = [tile(p) for p in Q]
    cnt = {}
    for t in tiles:
        cnt[t] = cnt.get(t, 0) + 1
    w = [dx * dy / cnt[t] for t in tiles]
    Z = [[1.0] + [float(v) for v in covariates(p[0], p[1])] for p in Q]
    k = len(Z[0])
    theta = [math.log(len(data) / ((x1 - x0) * (y1 - y0)))] + [0.0] * (k - 1)
    for _ in range(max_iter):
        eta = [ssum(Z[j][a] * theta[a] for a in range(k)) for j in range(len(Q))]
        mu = [math.exp(v) for v in eta]
        g = [
            ssum(Z[j][a] * (z[j] - w[j] * mu[j]) for j in range(len(Q))) - (ridge * theta[a] if a > 0 else 0.0)
            for a in range(k)
        ]
        H = [
            [
                ssum(w[j] * mu[j] * Z[j][a] * Z[j][b] for j in range(len(Q))) + (ridge if (a == b and a > 0) else 0.0)
                for b in range(k)
            ]
            for a in range(k)
        ]
        step = solve(H, g)
        theta = [a + b for a, b in zip(theta, step)]
        if max(abs(v) for v in step) <= 1e-12:
            break
    V = inverse(H)
    return RichResult(
        payload={"coefficients": theta, "se": [math.sqrt(V[a][a]) for a in range(k)], "weights": w, "n_quad": len(Q)}
    )


def st_k_function(points, r_values, t_values, area: float, time_length: float, *, method: str = "diggle") -> RichResult:
    r"""Space-time K and L functions (Diggle, Chetwynd, Haggkvist and Morris 1995), no edge correction.

    ``K(r, t) = |A| |T| / (n (n - 1)) sum_{i != j} 1(d_ij <= r) 1(|t_i - t_j| <= t)``
    (``2 pi r^2 t`` under complete space-time randomness) and
    ``L(r, t) = sqrt(K / (2 pi t))`` (``r`` under randomness). ``points``
    rows are ``(x, y, t)``.

    References
    ----------
    Diggle, P. J., Chetwynd, A. G., Haggkvist, R. and Morris, S. E. (1995).
    Second-order analysis of space-time clustering. Statistical Methods in
    Medical Research 4, 124-136. Gabriel, E. and Diggle, P. J. (2009).
    Second-order analysis of inhomogeneous spatio-temporal point process
    data. Statistica Neerlandica 63, 43-51.

    Examples
    --------
    >>> r = st_k_function([(0, 0, 0), (1, 0, 1), (0, 1, 5)], [1.0], [1.0], 4.0, 5.0)
    >>> round(r.K[0][0], 12)
    6.666666666667
    """
    P = [(float(a), float(b), float(c)) for a, b, c in points]
    n = len(P)
    K = []
    for r in r_values:
        row = []
        for t in t_values:
            c = 0
            for i in range(n):
                for j in range(n):
                    if i != j and math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) <= r and abs(P[i][2] - P[j][2]) <= t:
                        c += 1
            row.append(area * time_length * c / (n * n if method == "stpp" else n * (n - 1)))
        K.append(row)
    L = [
        [
            math.sqrt(K[a][b] / (2 * math.pi * t_values[b])) if t_values[b] > 0 else float("nan")
            for b in range(len(t_values))
        ]
        for a in range(len(r_values))
    ]
    return RichResult(payload={"K": K, "L": L})


def st_g_function(points, r_values, t_values) -> list:
    r"""Space-time nearest-neighbour distribution ``G(r, t)``: the share of events with another event within
    spatial distance ``r`` and time lag ``t`` (cylindrical neighbourhoods, no edge correction).

    References
    ----------
    Gabriel, E. (2014). Estimating second-order characteristics of
    inhomogeneous spatio-temporal point processes. Methodol. Comput. Appl.
    Probab. 16, 411-431. van Lieshout, M. N. M. (2011). A J-function for
    inhomogeneous point processes. Statistica Neerlandica 65, 183-201.

    Examples
    --------
    >>> st_g_function([(0, 0, 0), (1, 0, 1), (0, 1, 5)], [1.0], [1.0])
    [[0.6666666666666666]]
    """
    P = [(float(a), float(b), float(c)) for a, b, c in points]
    n = len(P)
    out = []
    for r in r_values:
        row = []
        for t in t_values:
            hit = sum(
                1
                for i in range(n)
                if any(
                    j != i and math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) <= r and abs(P[i][2] - P[j][2]) <= t
                    for j in range(n)
                )
            )
            row.append(hit / n)
        out.append(row)
    return out


def st_j_function(points, r_values, t_values, window, time_range, *, n_grid: int = 10, n_time: int = 10) -> RichResult:
    r"""Space-time J function ``J(r, t) = (1 - G(r, t)) / (1 - F(r, t))``.

    ``F`` is the empty-space function estimated on a regular grid of
    ``n_grid^2 x n_time`` reference locations (cell centres) in the window
    and time range; ``G`` as :func:`st_g_function`. ``J < 1`` indicates
    clustering, ``J > 1`` regularity (no edge correction).

    References
    ----------
    van Lieshout, M. N. M. and Baddeley, A. J. (1996). A nonparametric
    measure of spatial interaction in point patterns. Statistica Neerlandica
    50, 344-361. van Lieshout, M. N. M. (2011). Statistica Neerlandica 65, 183-201.

    Examples
    --------
    >>> r = st_j_function([(0.2, 0.2, 0.1), (0.8, 0.8, 0.9)], [0.1], [0.1], (0, 1, 0, 1), (0, 1), n_grid=2, n_time=2)
    >>> r.F[0][0], r.J[0][0]
    (0.0, 1.0)
    """
    x0, x1, y0, y1 = window
    s0, s1 = time_range
    ref = [
        (x0 + (i + 0.5) * (x1 - x0) / n_grid, y0 + (j + 0.5) * (y1 - y0) / n_grid, s0 + (k + 0.5) * (s1 - s0) / n_time)
        for k in range(n_time)
        for j in range(n_grid)
        for i in range(n_grid)
    ]
    P = [(float(a), float(b), float(c)) for a, b, c in points]
    G = st_g_function(P, r_values, t_values)
    F = []
    for r in r_values:
        row = []
        for t in t_values:
            hit = sum(
                1 for u in ref if any(math.hypot(u[0] - p[0], u[1] - p[1]) <= r and abs(u[2] - p[2]) <= t for p in P)
            )
            row.append(hit / len(ref))
        F.append(row)
    J = [
        [(1 - G[a][b]) / (1 - F[a][b]) if F[a][b] < 1 else float("nan") for b in range(len(t_values))]
        for a in range(len(r_values))
    ]
    return RichResult(payload={"G": G, "F": F, "J": J})


def cheatsheet() -> str:
    return (
        "area_interaction_simulate / thomas_simulate / thomas_k / lgcp_simulate_grid / abramson_intensity / "
        "berman_turner_fit / st_k_function / st_g_function / st_j_function -> point processes."
    )
