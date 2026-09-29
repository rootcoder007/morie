# morie.fn -- function file (rootcoder007/morie)
"""Cox (doubly stochastic) point processes: log-Gaussian Cox and Thomas cluster process simulation with
their first- and second-order moments, and Voronoi residuals of a fitted log-linear intensity."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._s03core import chol
from .krgsys import kriging_covariance
from .spsampling import _area, _clip

__all__ = ["lgcp_simulate", "lgcp_moments", "thomas_simulate", "thomas_pcf", "voronoi_residuals"]


class _U:
    """Sequential Philox draws (uniform or normal) in blocks of 4096 on consecutive streams."""

    def __init__(self, seed, stream0, normal=False):
        self.seed, self.block, self.buf, self.pos, self.normal = seed, stream0, [], 0, normal

    def next(self):
        if self.pos >= len(self.buf):
            gen = random_normal if self.normal else random_uniform
            self.buf = [float(v) for v in gen(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.pos = 0
        self.pos += 1
        return self.buf[self.pos - 1]


def _poisson(m, U):
    """Poisson draw by inversion of the cdf (sequential search)."""
    if m > 600:
        raise ValueError("Poisson mean too large for inversion")
    u = U.next()
    k, p = 0, math.exp(-m)
    F = p
    while u > F and p > 0:
        k += 1
        p *= m / k
        F += p
    return k


def lgcp_simulate(window, n_grid: int, mu: float, model, *, seed: int = 1) -> RichResult:
    r"""Simulate a log-Gaussian Cox process on a rectangle (Moller, Syversveen and Waagepetersen 1998).

    A Gaussian field ``Z`` with mean ``mu`` and covariance ``model``
    (:func:`morie.fn.krgsys.kriging_covariance`) is drawn at the centres of an
    ``n_grid x n_grid`` lattice by Cholesky factorisation of Philox normals;
    given ``Lambda = exp(Z)``, each cell receives a Poisson number of points
    with mean ``Lambda |cell|``, placed uniformly (Philox, streams 1000+).

    References
    ----------
    Moller, J., Syversveen, A. R. and Waagepetersen, R. P. (1998). Log
    Gaussian Cox processes. *Scandinavian Journal of Statistics*, 25, 451-482.

    Examples
    --------
    >>> r = lgcp_simulate((0, 1, 0, 1), 4, 3.0, {"model": "Exp", "psill": 0.5, "range": 0.2}, seed=2)
    >>> len(r.field), all(0 <= x <= 1 and 0 <= y <= 1 for x, y in r.points)
    (16, True)
    """
    x0, x1, y0, y1 = [float(v) for v in window]
    g = int(n_grid)
    dx, dy = (x1 - x0) / g, (y1 - y0) / g
    cen = [(x0 + (j + 0.5) * dx, y0 + (i + 0.5) * dy) for i in range(g) for j in range(g)]
    m = len(cen)
    C = [
        [kriging_covariance(math.dist(cen[a], cen[b]), model) + (1e-10 if a == b else 0.0) for b in range(m)]
        for a in range(m)
    ]
    L = chol(C)
    e = [float(v) for v in random_normal(m, seed=seed)]
    Z = [mu + ssum(L[a][k] * e[k] for k in range(a + 1)) for a in range(m)]
    U = _U(seed, 1000)
    pts = []
    for a in range(m):
        k = _poisson(math.exp(Z[a]) * dx * dy, U)
        for _ in range(k):
            pts.append((cen[a][0] - dx / 2 + dx * U.next(), cen[a][1] - dy / 2 + dy * U.next()))
    return RichResult(payload={"points": pts, "field": Z, "centres": cen})


def lgcp_moments(mu: float, sigma2: float, area: float, r: float = 0.0, model=None) -> RichResult:
    r"""Intensity ``exp(mu + sigma2/2)``, expected count and pair correlation ``g(r) = exp(C(r))`` of an LGCP.

    ``C`` is the covariance of the Gaussian field (``model``, whose sill should
    be ``sigma2``); without ``model`` only first-order moments are returned.

    References
    ----------
    Moller, Syversveen and Waagepetersen (1998), Theorem 1.

    Examples
    --------
    >>> r = lgcp_moments(1.0, 0.5, 2.0, 0.1, {"model": "Exp", "psill": 0.5, "range": 0.2})
    >>> round(r.intensity, 10), round(r.pcf, 10)
    (3.4903429575, 1.354273746)
    """
    lam = math.exp(mu + sigma2 / 2)
    out = {"intensity": lam, "expected_count": lam * area}
    if model is not None:
        out["pcf"] = math.exp(kriging_covariance(r, model))
    return RichResult(payload=out)


def thomas_simulate(kappa: float, scale: float, mu: float, window, *, seed: int = 1) -> RichResult:
    r"""Simulate a (modified) Thomas cluster process, a Cox process with shot-noise intensity.

    Parents are Poisson with intensity ``kappa`` on the window dilated by
    ``4 scale`` (to avoid edge effects), each has a Poisson(``mu``) number of
    offspring displaced by isotropic N(0, ``scale^2`` I) vectors, and
    offspring outside the window are discarded (Thomas 1949; Diggle 2013).
    Uniforms come from Philox streams 0, 1, ..., normals from streams 5000, 5001, ....

    References
    ----------
    Thomas, M. (1949). A generalization of Poisson's binomial limit for use in
    ecology. *Biometrika*, 36, 18-25.
    Diggle, P. J. (2013). *Statistical Analysis of Spatial and Spatio-Temporal
    Point Patterns*, 3rd edn. CRC Press, section 5.3.

    Examples
    --------
    >>> r = thomas_simulate(10, 0.05, 5, (0, 1, 0, 1), seed=3)
    >>> all(0 <= x <= 1 and 0 <= y <= 1 for x, y in r.points)
    True
    """
    x0, x1, y0, y1 = [float(v) for v in window]
    d = 4 * scale
    ex0, ex1, ey0, ey1 = x0 - d, x1 + d, y0 - d, y1 + d
    U = _U(seed, 0)
    npar = _poisson(kappa * (ex1 - ex0) * (ey1 - ey0), U)
    par = [(ex0 + (ex1 - ex0) * U.next(), ey0 + (ey1 - ey0) * U.next()) for _ in range(npar)]
    Z = _U(seed, 5000, normal=True)
    pts = []
    for px, py in par:
        k = _poisson(mu, U)
        for _ in range(k):
            x = px + scale * Z.next()
            y = py + scale * Z.next()
            if x0 <= x <= x1 and y0 <= y <= y1:
                pts.append((x, y))
    return RichResult(payload={"points": pts, "parents": par})


def thomas_pcf(r: float, kappa: float, scale: float) -> RichResult:
    r"""Pair correlation ``g(r) = 1 + exp(-r^2 / (4 s^2)) / (4 pi kappa s^2)`` and ``K(r) = pi r^2 + (1 - exp(-r^2/(4 s^2))) / kappa`` of the Thomas process.

    References
    ----------
    Diggle (2013), section 5.3; Moller, J. and Waagepetersen, R. P. (2004).
    *Statistical Inference and Simulation for Spatial Point Processes*. Chapman and Hall, section 5.3.

    Examples
    --------
    >>> r = thomas_pcf(0.1, 10.0, 0.05)
    >>> round(r.pcf, 10), round(r.K, 10)
    (2.1709966305, 0.0946279824)
    """
    s2 = scale * scale
    return RichResult(
        payload={
            "pcf": 1 + math.exp(-r * r / (4 * s2)) / (4 * math.pi * kappa * s2),
            "K": math.pi * r * r + (1 - math.exp(-r * r / (4 * s2))) / kappa,
        }
    )


_DUN = [
    (0.225, 1 / 3, 1 / 3, 1 / 3),
    (0.132394152788506, 0.059715871789770, 0.470142064105115, 0.470142064105115),
    (0.132394152788506, 0.470142064105115, 0.059715871789770, 0.470142064105115),
    (0.132394152788506, 0.470142064105115, 0.470142064105115, 0.059715871789770),
    (0.125939180544827, 0.797426985353087, 0.101286507323456, 0.101286507323456),
    (0.125939180544827, 0.101286507323456, 0.797426985353087, 0.101286507323456),
    (0.125939180544827, 0.101286507323456, 0.101286507323456, 0.797426985353087),
]


def _cell_integral(cell, beta):
    s = 0.0
    a = cell[0]
    for i in range(1, len(cell) - 1):
        b, c = cell[i], cell[i + 1]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        q = 0.0
        for w, l1, l2, l3 in _DUN:
            x = l1 * a[0] + l2 * b[0] + l3 * c[0]
            y = l1 * a[1] + l2 * b[1] + l3 * c[1]
            q += w * math.exp(beta[0] + beta[1] * x + beta[2] * y)
        s += area * q
    return s


def voronoi_residuals(points, bbox, beta) -> RichResult:
    r"""Voronoi residuals of a point process with log-linear intensity ``exp(b0 + b1 x + b2 y)``.

    Each point's Voronoi cell ``V_i`` (within the box) has raw residual
    ``1 - int_{V_i} lambda(u) du`` (Bray, Wong, Barr and Schoenberg 2014): the
    cells adapt to the local point density, so the residuals are close to
    symmetric where grid residuals are skewed. The integral uses a fan
    triangulation and the 7-point degree-5 Dunavant rule on each triangle.

    References
    ----------
    Bray, A., Wong, K., Barr, C. D. and Schoenberg, F. P. (2014). Voronoi
    residual analysis of spatial point process models with applications to
    California earthquake forecasts. *Annals of Applied Statistics*, 8, 2247-2267.
    Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian
    quadrature rules for the triangle. *IJNME*, 21, 1129-1148.

    Examples
    --------
    >>> r = voronoi_residuals([(0.25, 0.5), (0.75, 0.5)], (0, 0, 1, 1), [math.log(2.0), 0.0, 0.0])
    >>> [round(v, 12) for v in r.residuals]
    [0.0, 0.0]
    """
    P = [(float(a), float(b)) for a, b in points]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    b = [float(v) for v in beta]
    res, areas = [], []
    for i, (px, py) in enumerate(P):
        cell = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for j, (qx, qy) in enumerate(P):
            if j == i or (qx == px and qy == py):
                continue
            cell = _clip(cell, qx - px, qy - py, 0.5 * (qx * qx + qy * qy - px * px - py * py))
            if not cell:
                break
        areas.append(_area(cell) if len(cell) >= 3 else 0.0)
        res.append(1.0 - (_cell_integral(cell, b) if len(cell) >= 3 else 0.0))
    return RichResult(payload={"residuals": res, "areas": areas})


def cheatsheet() -> str:
    return "lgcp_simulate / lgcp_moments / thomas_simulate / thomas_pcf / voronoi_residuals -> Cox processes."

# alias kept from the retired placeholder of the same name
log_gaussian_cox = lgcp_simulate

# alias kept from the retired placeholder of the same name
pp_delaunay_resid = voronoi_residuals
