# morie.fn -- function file (rootcoder007/morie)
"""Spatial simulation and interpolation: turning bands with the spherical covariance (Matheron dilution),
diffusion-limited aggregation, moving least squares, deformation-based nonstationary Gaussian fields and
simulated-annealing simulation matching a variogram."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._s03core import chol
from .krgsys import kriging_covariance

__all__ = [
    "turning_bands_spherical",
    "dla_aggregate",
    "moving_least_squares",
    "deformation_field_simulate",
    "annealing_simulation",
]


class _U:
    def __init__(self, seed, stream0):
        self.seed, self.block, self.buf, self.pos = seed, stream0, [], 0

    def next(self):
        if self.pos >= len(self.buf):
            self.buf = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def turning_bands_spherical(
    coords, *, sill: float = 1.0, range_: float = 1.0, n_bands: int = 100, lam=None, seed: int = 1
) -> RichResult:
    r"""Gaussian random field with the spherical covariance by turning bands in three dimensions.

    The spherical covariance ``sill (1 - 3h/2a + h^3/2a^3)`` in ``R^3``
    corresponds on each line to ``1 - 3h/a + 2h^3/a^3``, the covariance of
    Matheron's dilution of ``g(t) = t 1{|t| < a/2}`` by a Poisson process
    (intensity ``lam``, default ``50 / a``) with random signs; its variance
    is ``lam a^3 / 12``. Bands point along a Fibonacci lattice on the sphere
    and the field is ``sqrt(sill / L) sum_l z_l(<x, u_l>)`` normalised to
    unit line variance. Two-dimensional coordinates are embedded in the plane ``z = 0``.

    References
    ----------
    Matheron, G. (1973). The intrinsic random functions and their
    applications. *Advances in Applied Probability*, 5, 439-468.
    Lantuejoul, C. (2002). *Geostatistical Simulation: Models and
    Algorithms*. Springer, ch. 15 (dilution functions of the turning bands).

    Examples
    --------
    >>> r = turning_bands_spherical([(0, 0, 0), (0.5, 0, 0), (5, 5, 5)], n_bands=10, seed=2)
    >>> len(r.field)
    3
    """
    X = [tuple(float(v) for v in c) + (0.0,) * (3 - len(c)) for c in coords]
    a = float(range_)
    lam = 50.0 / a if lam is None else float(lam)
    L = int(n_bands)
    ga = math.pi * (3 - math.sqrt(5))
    dirs = []
    for band in range(L):
        zc = 1 - (2 * band + 1) / L
        r = math.sqrt(max(1 - zc * zc, 0.0))
        dirs.append((r * math.cos(band * ga), r * math.sin(band * ga), zc))
    U = _U(seed, 0)
    field = [0.0] * len(X)
    norm = math.sqrt(sill / L) / math.sqrt(lam * a**3 / 12.0)
    for u in dirs:
        t = [u[0] * p[0] + u[1] * p[1] + u[2] * p[2] for p in X]
        lo, hi = min(t) - a / 2, max(t) + a / 2
        s = lo
        pts = []
        while True:
            s += -math.log(1.0 - U.next()) / lam
            if s > hi:
                break
            pts.append((s, 1.0 if U.next() < 0.5 else -1.0))
        for i, ti in enumerate(t):
            v = 0.0
            for tk, e in pts:
                d = ti - tk
                if -a / 2 < d < a / 2:
                    v += e * d
            field[i] += norm * v
    return RichResult(payload={"field": field, "directions": dirs})


def dla_aggregate(n_particles: int, *, seed: int = 1) -> RichResult:
    r"""Diffusion-limited aggregation on the square lattice (Witten and Sander 1981).

    A seed occupies the origin; each walker starts on the circle of radius
    ``R_max + 5`` (``R_max`` the current cluster radius, Philox angle), makes
    nearest-neighbour random steps, sticks when a neighbour is occupied and
    is relaunched if it wanders beyond ``3 (R_max + 5)``. Returns the sites
    in order of attachment, the radius of gyration as the cluster grows and
    the mass-radius fractal dimension (slope of ``log N`` on ``log R_g`` over
    the second half of the growth; about 1.71 in two dimensions).

    References
    ----------
    Witten, T. A. and Sander, L. M. (1981). Diffusion-limited aggregation, a
    kinetic critical phenomenon. *Physical Review Letters*, 47, 1400-1403.

    Examples
    --------
    >>> r = dla_aggregate(30, seed=4)
    >>> len(r.sites), r.sites[0]
    (31, (0, 0))
    """
    occ = {(0, 0)}
    sites = [(0, 0)]
    U = _U(seed, 0)
    rmax = 0.0
    rg = [0.0]
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(sites) <= n_particles:
        R = rmax + 5.0
        th = 2 * math.pi * U.next()
        x, y = int(round(R * math.cos(th))), int(round(R * math.sin(th)))
        while True:
            if any((x + dx, y + dy) in occ for dx, dy in moves):
                if (x, y) not in occ:
                    occ.add((x, y))
                    sites.append((x, y))
                    rmax = max(rmax, math.hypot(x, y))
                    mx = ssum(p[0] for p in sites) / len(sites)
                    my = ssum(p[1] for p in sites) / len(sites)
                    rg.append(math.sqrt(ssum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in sites) / len(sites)))
                break
            dx, dy = moves[min(int(U.next() * 4), 3)]
            x, y = x + dx, y + dy
            if math.hypot(x, y) > 3 * R:
                th = 2 * math.pi * U.next()
                x, y = int(round(R * math.cos(th))), int(round(R * math.sin(th)))
    half = [(math.log(rg[k]), math.log(k + 1)) for k in range(len(rg) // 2, len(rg)) if rg[k] > 0]
    mx = ssum(p[0] for p in half) / len(half)
    my = ssum(p[1] for p in half) / len(half)
    D = ssum((p[0] - mx) * (p[1] - my) for p in half) / ssum((p[0] - mx) ** 2 for p in half)
    return RichResult(payload={"sites": sites, "radius_of_gyration": rg, "fractal_dimension": D})


def moving_least_squares(points, values, targets, h: float, *, degree: int = 1) -> RichResult:
    r"""Moving least-squares approximation (Lancaster and Salkauskas 1981) with Gaussian weights.

    At each target ``x`` a polynomial of the given ``degree`` (1: linear,
    2: quadratic in the coordinates) is fitted by weighted least squares with
    weights ``exp(-||x_i - x||^2 / h^2)`` and evaluated at ``x``; the result is
    smooth and reproduces polynomials of that degree exactly.

    References
    ----------
    Lancaster, P. and Salkauskas, K. (1981). Surfaces generated by moving
    least squares methods. *Mathematics of Computation*, 37, 141-158.

    Examples
    --------
    >>> P = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    >>> v = [1 + 2 * x - y for x, y in P]
    >>> round(moving_least_squares(P, v, [(0.3, 0.7)], 0.8).prediction[0], 12)
    0.9
    """
    P = [tuple(float(v) for v in p) for p in points]
    V = [float(v) for v in values]
    d = len(P[0])

    def basis(p):
        b = [1.0] + list(p)
        if degree == 2:
            b += [p[i] * p[j] for i in range(d) for j in range(i, d)]
        return b

    B = [basis(p) for p in P]
    m = len(B[0])
    out = []
    for t in targets:
        t = tuple(float(v) for v in t)
        w = [math.exp(-ssum((a - b) ** 2 for a, b in zip(p, t)) / (h * h)) for p in P]
        M = [[ssum(w[i] * B[i][a] * B[i][b] for i in range(len(P))) for b in range(m)] for a in range(m)]
        r = [ssum(w[i] * B[i][a] * V[i] for i in range(len(P))) for a in range(m)]
        coef = solve(M, r)
        bt = basis(t)
        out.append(ssum(bt[k] * coef[k] for k in range(m)))
    return RichResult(payload={"prediction": out})


def deformation_field_simulate(coords, deformed, model, *, nsim: int = 1, seed: int = 1) -> RichResult:
    r"""Nonstationary Gaussian field by spatial deformation (Sampson and Guttorp 1992).

    ``cov(Z(s), Z(s')) = C(||f(s) - f(s')||)`` with a stationary isotropic
    covariance ``model`` (:func:`morie.fn.krgsys.kriging_covariance`) on the
    deformed ("D-plane") coordinates ``f(s)`` given in ``deformed``; the
    field on the geographic ``coords`` is anisotropic and nonstationary.
    Realisations are ``L e`` with ``L`` the Cholesky factor and Philox
    normals (stream ``k`` for realisation ``k``).

    References
    ----------
    Sampson, P. D. and Guttorp, P. (1992). Nonparametric estimation of
    nonstationary spatial covariance structure. *JASA*, 87, 108-119.

    Examples
    --------
    >>> r = deformation_field_simulate([(0, 0), (1, 0)], [(0, 0), (3, 0)], {"model": "Exp", "psill": 1.0, "range": 1.0})
    >>> round(r.covariance[0][1], 12)
    0.049787068368
    """
    G = [tuple(float(v) for v in p) for p in deformed]
    n = len(G)
    if len(coords) != n:
        raise ValueError("coords and deformed must have the same length")
    C = [
        [kriging_covariance(math.dist(G[i], G[j]), model) + (1e-12 if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]
    L = chol(C)
    sims = []
    for k in range(nsim):
        e = [float(v) for v in random_normal(n, seed=seed, stream=k)]
        sims.append([ssum(L[i][j] * e[j] for j in range(i + 1)) for i in range(n)])
    return RichResult(payload={"realisations": sims, "covariance": C})


def annealing_simulation(
    nx: int, ny: int, values, model, lags, *, n_iter: int = 5000, t0: float = 1.0, cooling: float = 0.001, seed: int = 1
) -> RichResult:
    r"""Simulated-annealing simulation of a grid honouring a histogram and a variogram (Deutsch and Journel's ``sasim``).

    The grid starts as a Philox permutation of ``values`` (``nx * ny`` of
    them, so the histogram is exact) and pairs of cells are swapped; a swap
    is kept when it lowers the objective ``O = sum_h ((g*(h) - g(h)) / g(h))^2``
    over the lag vectors ``lags`` (``g*`` the grid's semivariogram, ``g`` the
    model: ``C(0) - C(h)`` from :func:`morie.fn.krgsys.kriging_covariance`) or
    with probability ``exp(-dO / T)``, ``T = t0 / (1 + k * cooling)``.

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide*, 2nd edn. Oxford University Press, section V.6.
    Kirkpatrick, S., Gelatt, C. D. and Vecchi, M. P. (1983). Optimization by
    simulated annealing. *Science*, 220, 671-680.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 3.0}
    >>> r = annealing_simulation(6, 6, [float(i % 7) for i in range(36)], m, [(1, 0), (0, 1)], n_iter=200)
    >>> r.objective[-1] <= r.objective[0]
    True
    """
    N = nx * ny
    vals = [float(v) for v in values]
    if len(vals) != N:
        raise ValueError("need nx * ny values")
    U = _U(seed, 0)
    idx = list(range(N))
    for k in range(N - 1, 0, -1):  # Fisher-Yates
        j = min(int(U.next() * (k + 1)), k)
        idx[k], idx[j] = idx[j], idx[k]
    g = [vals[i] for i in idx]
    c0 = kriging_covariance(0.0, model)
    target = [c0 - kriging_covariance(math.hypot(dx, dy), model) for dx, dy in lags]
    pairs = []
    for dx, dy in lags:
        pl = []
        for y in range(ny):
            for x in range(nx):
                if 0 <= x + dx < nx and 0 <= y + dy < ny:
                    pl.append((y * nx + x, (y + dy) * nx + x + dx))
        pairs.append(pl)
    sums = [ssum((g[a] - g[b]) ** 2 for a, b in pl) for pl in pairs]

    def obj(s):
        return ssum(((s[k] / (2 * len(pairs[k])) - target[k]) / target[k]) ** 2 for k in range(len(pairs)))

    inv = [[[] for _ in range(N)] for _ in pairs]
    for k, pl in enumerate(pairs):
        for a, b in pl:
            inv[k][a].append((a, b))
            inv[k][b].append((a, b))
    cur = obj(sums)
    path = [cur]
    for it in range(n_iter):
        i = min(int(U.next() * N), N - 1)
        j = min(int(U.next() * N), N - 1)
        u = U.next()
        if i == j or g[i] == g[j]:
            path.append(cur)
            continue
        touched = [sorted(set(inv[k][i] + inv[k][j])) for k in range(len(pairs))]
        old = [ssum((g[a] - g[b]) ** 2 for a, b in touched[k]) for k in range(len(pairs))]
        g[i], g[j] = g[j], g[i]
        new = [ssum((g[a] - g[b]) ** 2 for a, b in touched[k]) for k in range(len(pairs))]
        cand = [sums[k] - old[k] + new[k] for k in range(len(pairs))]
        o = obj(cand)
        T = t0 / (1.0 + it * cooling)
        if o <= cur or u < math.exp(-(o - cur) / T):
            sums, cur = cand, o
        else:
            g[i], g[j] = g[j], g[i]
        path.append(cur)
    return RichResult(
        payload={"grid": [g[y * nx : (y + 1) * nx] for y in range(ny)], "objective": path, "target": target}
    )


def cheatsheet() -> str:
    return (
        "turning_bands_spherical / dla_aggregate / moving_least_squares / deformation_field_simulate / "
        "annealing_simulation -> spatial simulation."
    )
