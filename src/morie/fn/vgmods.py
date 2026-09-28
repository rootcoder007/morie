# morie.fn -- function file (rootcoder007/morie)
"""Variogram models, sample variogram estimators, variogram maps and weighted least squares fitting (gstat conventions)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._sci_core import kv

__all__ = [
    "vgm_semivariance",
    "vgm_covariance",
    "vgm_correlogram",
    "vgm_spectral_density",
    "sample_variogram",
    "variogram_map",
    "fit_variogram",
    "variogram_loglik",
    "likfit",
    "select_variogram_model",
]

_BOUNDED = (
    "Nug",
    "Exp",
    "Sph",
    "Gau",
    "Exc",
    "Mat",
    "Ste",
    "Cir",
    "Lin",
    "Bes",
    "Pen",
    "Wav",
    "Hol",
    "Cub",
    "Cau",
    "JBes",
    "Dmp",
)
_MODELS = _BOUNDED + ("Per", "Cos", "Log", "Pow")


def _comps(model):
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _jbes_rho(x, nu):
    """Gamma(nu + 1) (2/x)^nu J_nu(x) by Poisson's integral with composite 16-point Gauss-Legendre."""
    from .krgsys import _gauss_legendre

    gx, gw = _gauss_legendre(16)
    m = int(math.ceil(x)) + 8
    h = math.pi / m
    s = 0.0
    for p in range(m):
        for t, w in zip(gx, gw):
            th = h * (p + 0.5 * (t + 1.0))
            s += 0.5 * h * w * math.cos(x * math.cos(th)) * math.sin(th) ** (2.0 * nu)
    return math.gamma(nu + 1.0) / (math.sqrt(math.pi) * math.gamma(nu + 0.5)) * s


def _matern_rho(r, k):
    """Matern correlation; closed form exp(-r) p!/(2p)! sum (p+i)!/(i!(p-i)!) (2r)^(p-i) for k = p + 1/2."""
    p = k - 0.5
    if p == int(p) and 0 <= p <= 20:
        p = int(p)
        f = math.factorial
        return (
            math.exp(-r)
            * f(p)
            / f(2 * p)
            * sum(f(p + i) / (f(i) * f(p - i)) * (2.0 * r) ** (p - i) for i in range(p + 1))
        )
    return 2.0 ** (1.0 - k) / math.gamma(k) * r**k * float(kv(k, r))


def _unit(h, c):
    """Unit-sill semivariance of one component at distance ``h > 0``."""
    m = c.get("model", "Exp")
    a = float(c.get("range", 1.0))
    k = float(c.get("kappa", 0.5))
    r = h / a if a > 0 else h
    if m == "Nug":
        return 1.0
    if m == "Exp":
        return 1.0 - math.exp(-r)
    if m == "Sph":
        return 1.5 * r - 0.5 * r**3 if r < 1 else 1.0
    if m == "Gau":
        return 1.0 - math.exp(-r * r)
    if m == "Exc":
        return 1.0 - math.exp(-(r**k))
    if m == "Mat":
        return 1.0 - _matern_rho(r, k)
    if m == "Ste":
        return 1.0 - _matern_rho(2.0 * math.sqrt(k) * r, k)
    if m == "Cir":
        return (2.0 / math.pi) * (r * math.sqrt(1.0 - r * r) + math.asin(r)) if r < 1 else 1.0
    if m == "Lin":
        return (r if r < 1 else 1.0) if a > 0 else h
    if m == "Bes":
        return 1.0 - r * float(kv(1.0, r))
    if m == "Pen":
        return 15.0 / 8.0 * r - 5.0 / 4.0 * r**3 + 3.0 / 8.0 * r**5 if r < 1 else 1.0
    if m == "Wav":
        return 1.0 - math.sin(math.pi * r) / (math.pi * r)
    if m == "Hol":
        return 1.0 - math.sin(r) / r
    if m == "Cub":
        return 7.0 * r**2 - 35.0 / 4.0 * r**3 + 7.0 / 2.0 * r**5 - 3.0 / 4.0 * r**7 if r < 1 else 1.0
    if m == "Cau":
        return 1.0 - (1.0 + r * r) ** (-k)
    if m == "JBes":
        return 1.0 - _jbes_rho(r, k)
    if m == "Dmp":
        return 1.0 - math.exp(-r) * math.cos(h / float(c.get("period", 1.0)))
    if m == "Per":
        return 1.0 - math.cos(2.0 * math.pi * r)
    if m == "Cos":
        return 1.0 - math.cos(r)
    if m == "Log":
        return math.log(h + a)
    if m == "Pow":
        return h**a
    raise ValueError(f"model must be one of {_MODELS}")


def vgm_semivariance(h, model):
    r"""Semivariance of a (nested) variogram model at distance(s) ``h``.

    Components are dicts with ``model``, ``psill``, ``range`` (``kappa``,
    ``period`` where used) and an optional ``nugget``; ``gamma(0) = 0`` and
    ``gamma(h) = sum psill g(h)`` with the unit-sill shapes of
    ``gstat::vgm`` -- ``Nug``, ``Exp``, ``Sph``, ``Gau``, ``Exc`` (stable
    ``1 - exp(-(h/a)^kappa)``), ``Mat``, ``Ste`` (Matern in Stein's
    parameterisation), ``Cir``, ``Lin``, ``Bes`` (``1 - (h/a) K_1(h/a)``,
    Whittle), ``Pen``, ``Per`` (``1 - cos(2 pi h/a)``), ``Wav`` (``1 -
    sin(pi h/a)/(pi h/a)``), ``Hol`` (``1 - a sin(h/a)/h``), ``Log``
    (``log(h + a)``), ``Pow`` (``h^a``) -- plus ``Cub`` (cubic), ``Cau``
    (Cauchy ``1 - (1 + (h/a)^2)^{-kappa}``), as ``geoR::cov.spatial``,
    ``JBes`` (``1 - Gamma(kappa + 1)(2a/h)^kappa J_kappa(h/a)``), ``Dmp``
    (damped oscillation ``1 - exp(-h/a) cos(h/period)``) and ``Cos``
    (``1 - cos(h/a)``) (Chiles and Delfiner 2012, chapter 2).

    References
    ----------
    Chiles, J.-P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
    Uncertainty*, 2nd edn. Wiley, Hoboken.
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat
    package. *Computers and Geosciences*, 30(7), 683-691.

    Examples
    --------
    >>> round(vgm_semivariance(0.9, {"model": "Pen", "psill": 1.0, "range": 2.0}), 6)
    0.736764
    """
    comps = _comps(model)
    for c in comps:
        if c.get("model", "Exp") not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}")

    def one(x):
        x = abs(float(x))
        if x == 0.0:
            return 0.0
        return ssum(float(c.get("psill", 1.0)) * _unit(x, c) + float(c.get("nugget", 0.0)) for c in comps)

    if isinstance(h, (int, float)):
        return one(h)
    return [one(x) for x in h]


def _sill(model):
    comps = _comps(model)
    if any(c.get("model", "Exp") not in _BOUNDED for c in comps):
        raise ValueError("the model is unbounded: it has no covariance")
    return ssum(float(c.get("psill", 1.0)) + float(c.get("nugget", 0.0)) for c in comps)


def vgm_covariance(h, model):
    r"""Covariance ``C(h) = C(0) - gamma(h)`` of a bounded variogram model (``C(0)`` the total sill).

    Examples
    --------
    >>> round(vgm_covariance(0.9, {"model": "Cub", "psill": 2.0, "range": 2.0}), 6)
    0.636123
    """
    s = _sill(model)
    g = vgm_semivariance(h, model)
    return s - g if isinstance(g, float) else [s - v for v in g]


def vgm_correlogram(h, model):
    r"""Correlogram ``rho(h) = 1 - gamma(h) / C(0)`` of a bounded variogram model.

    Examples
    --------
    >>> round(vgm_correlogram(1.0, {"model": "Exp", "psill": 3.0, "range": 1.0}), 6)
    0.367879
    """
    s = _sill(model)
    g = vgm_semivariance(h, model)
    return 1.0 - g / s if isinstance(g, float) else [1.0 - v / s for v in g]


def vgm_spectral_density(omega: float, model: dict, d: int = 2) -> float:
    r"""Spectral density ``f`` with ``C(h) = int exp(i omega'h) f(omega) d omega`` in ``R^d``.

    Matern (``Exp`` is ``kappa = 1/2``) with sill ``s`` and range ``a``:
    ``s Gamma(kappa + d/2) a^d / (Gamma(kappa) pi^{d/2} (1 + a^2 |omega|^2)^{kappa + d/2})``;
    Gaussian ``exp(-(h/a)^2)``: ``s (a / (2 sqrt(pi)))^d exp(-a^2 |omega|^2 / 4)``
    (Stein 1999, section 2.10).

    References
    ----------
    Stein, M. L. (1999). *Interpolation of Spatial Data*. Springer, New York.

    Examples
    --------
    >>> round(vgm_spectral_density(0.0, {"model": "Exp", "psill": 1.0, "range": 2.0}, d=1), 6)
    0.63662
    """
    m = model.get("model", "Exp")
    s, a = float(model.get("psill", 1.0)), float(model.get("range", 1.0))
    w = abs(float(omega))
    if m in ("Exp", "Mat"):
        k = 0.5 if m == "Exp" else float(model.get("kappa", 0.5))
        return (
            s
            * math.gamma(k + d / 2.0)
            * a**d
            / (math.gamma(k) * math.pi ** (d / 2.0) * (1.0 + a * a * w * w) ** (k + d / 2.0))
        )
    if m == "Gau":
        return s * (a / (2.0 * math.sqrt(math.pi))) ** d * math.exp(-a * a * w * w / 4.0)
    raise ValueError("vgm_spectral_density supports Exp, Mat and Gau")


def _pairs(P):
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def _estimate(dz, est, zi=None, zj=None):
    N = len(dz)
    if est == "classical":
        return ssum(v * v for v in dz) / (2.0 * N)
    if est == "cressie":
        return (ssum(math.sqrt(abs(v)) for v in dz) / N) ** 4 / (0.457 + 0.494 / N) / 2.0
    if est == "mad":
        a = sorted(abs(v) for v in dz)
        med = a[N // 2] if N % 2 else 0.5 * (a[N // 2 - 1] + a[N // 2])
        return 1.099 * med * med
    if est == "pairwise_relative":
        return ssum((v / ((x + y) / 2.0)) ** 2 for v, x, y in zip(dz, zi, zj)) / (2.0 * N)
    if est == "relative":
        m = ssum(list(zi) + list(zj)) / (2.0 * N)
        return ssum(v * v for v in dz) / (2.0 * N) / (m * m)
    raise ValueError("estimator must be classical, cressie, mad, pairwise_relative or relative")


def sample_variogram(
    z,
    coords,
    *,
    cutoff: float | None = None,
    width: float | None = None,
    boundaries=None,
    alpha=None,
    tol_hor: float | None = None,
    estimator: str = "classical",
    covariogram: bool = False,
    cloud: bool = False,
    z2=None,
    threshold: float | None = None,
) -> RichResult:
    r"""Sample (semi)variogram of 2-D data, as ``gstat::variogram``.

    Pairs are binned by distance into ``(b_k, b_{k+1}]`` with boundaries
    ``boundaries``, or ``0, width, 2 width, ..`` up to ``cutoff`` (defaults:
    one third of the bounding-box diagonal, ``cutoff / 15``).  For each bin:
    ``np`` pairs, ``dist`` their mean distance and ``gamma`` from the
    estimator: ``classical`` (Matheron) ``sum dz^2 / (2 N)``; ``cressie``
    (Cressie and Hawkins 1980) ``(mean |dz|^{1/2})^4 / (0.457 + 0.494/N) /
    2``; ``mad`` (Dowd 1984) ``1.099 median(|dz|)^2``; ``pairwise_relative``
    ``sum (dz / ((z_i + z_j)/2))^2 / (2N)`` and ``relative`` (the classical
    value over the squared lag mean; Isaaks and Srivastava 1989).
    ``covariogram``: ``sum (z_i - m)(z_j - m) / N`` with ``m`` the mean,
    followed by gstat's lag-zero row (``np = n``, the variance with divisor
    ``n``).
    ``z2`` gives the pseudo cross-variogram ``sum (z_i - z2_j)^2 / (2N)``
    over ordered pairs (Myers 1991); ``threshold`` the indicator variogram
    of ``1(z <= threshold)``.  Directions ``alpha`` (degrees clockwise from
    north, as gstat) keep pairs whose axial angle lies within ``tol_hor``
    (default ``90 / len(alpha)``).  ``cloud`` returns every pair.

    References
    ----------
    Cressie, N. and Hawkins, D. M. (1980). Robust estimation of the
    variogram: I. *Mathematical Geology*, 12(2), 115-125.
    Dowd, P. A. (1984). The variogram and kriging: robust and resistant
    estimators. In *Geostatistics for Natural Resources Characterization*,
    91-106. Reidel, Dordrecht.
    Isaaks, E. H. and Srivastava, R. M. (1989). *An Introduction to Applied
    Geostatistics*. Oxford University Press, New York.
    Myers, D. E. (1991). Pseudo-cross variograms, positive-definiteness, and
    cokriging. *Mathematical Geology*, 23(6), 805-816.

    Examples
    --------
    >>> v = sample_variogram([1.0, 2.0, 4.0, 3.0], [(0, 0), (1, 0), (0, 1), (1, 1)], boundaries=[0, 1.2, 1.5])
    >>> v.np, [round(g, 6) for g in v.gamma]
    ([4, 2], [1.5, 2.0])
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    n = len(zv)
    if len(P) != n:
        raise ValueError("coords must match z")
    if threshold is not None:
        zv = [1.0 if v <= threshold else 0.0 for v in zv]
    z2v = None if z2 is None else [float(v) for v in np.asarray(z2, dtype=float).tolist()]
    if boundaries is None:
        if cutoff is None:
            xs, ys = [p[0] for p in P], [p[1] for p in P]
            cutoff = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 3.0
        if width is None:
            width = cutoff / 15.0
        B = [k * width for k in range(int(math.floor(cutoff / width + 1e-9)) + 1)]
        if B[-1] < cutoff - 1e-12:
            B.append(cutoff)
    else:
        B = [float(v) for v in boundaries]
    dirs = [None] if alpha is None else [float(a) for a in (alpha if isinstance(alpha, (list, tuple)) else [alpha])]
    tol = 90.0 / len(dirs) if tol_hor is None else float(tol_hor)
    zm = ssum(zv) / n
    idx = [(i, j) for i in range(n) for j in range(n) if i != j] if z2v is not None else list(_pairs(P))
    if cloud:
        out = [
            (i, j, math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]), 0.5 * (zv[i] - (z2v[j] if z2v else zv[j])) ** 2)
            for i, j in idx
        ]
        return RichResult(
            payload={
                "left": [o[0] for o in out],
                "right": [o[1] for o in out],
                "dist": [o[2] for o in out],
                "gamma": [o[3] for o in out],
            }
        )
    G, NP, D, H = [], [], [], []
    for a in dirs:
        for k in range(len(B) - 1):
            dz, zi, zj, ds = [], [], [], []
            for i, j in idx:
                dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
                d = math.hypot(dx, dy)
                if not (B[k] < d <= B[k + 1]) and not (k == 0 and d == 0.0 and B[0] == 0.0):
                    continue
                if a is not None:
                    ang = math.degrees(math.atan2(dx, dy)) % 180.0
                    diff = abs(ang - a % 180.0)
                    if min(diff, 180.0 - diff) > tol:
                        continue
                if covariogram:
                    dz.append((zv[i] - zm) * (zv[j] - zm))
                else:
                    dz.append(zv[i] - (z2v[j] if z2v else zv[j]))
                zi.append(zv[i])
                zj.append(zv[j])
                ds.append(d)
            if not dz:
                continue
            G.append(ssum(dz) / len(dz) if covariogram else _estimate(dz, estimator, zi, zj))
            NP.append(len(dz))
            D.append(ssum(ds) / len(ds))
            H.append(0.0 if a is None else a)
    if covariogram:
        # gstat appends the lag-zero covariance (the variance with divisor n)
        NP.append(n)
        D.append(0.0)
        G.append(ssum((v - zm) ** 2 for v in zv) / n)
        H.append(0.0)
    return RichResult(payload={"np": NP, "dist": D, "gamma": G, "dir_hor": H})


def variogram_map(z, coords, *, cutoff: float, width: float) -> RichResult:
    r"""Variogram map: semivariance on a grid of lag vectors ``(dx, dy)``.

    Every ordered pair contributes its lag ``s_j - s_i`` (so the map is
    point-symmetric) to the square cell of side ``width`` containing it, for
    cells centred at ``k width``, ``|k| width <= cutoff``; ``gamma`` is
    ``sum dz^2 / (2N)`` per non-empty cell (Isaaks and Srivastava 1989,
    chapter 7; ``gstat::variogram(map = TRUE)`` layout).

    Examples
    --------
    >>> r = variogram_map([1.0, 2.0, 4.0], [(0, 0), (1, 0), (0, 1)], cutoff=1.0, width=1.0)
    >>> [(x, y, g) for x, y, g in zip(r.dx, r.dy, r.gamma) if x == 1.0]
    [(1.0, -1.0, 2.0), (1.0, 0.0, 0.5)]
    """
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    K = int(math.floor(cutoff / width + 1e-9))
    acc = {}
    for i in range(len(zv)):
        for j in range(len(zv)):
            if i == j:
                continue
            cx = round((P[j][0] - P[i][0]) / width)
            cy = round((P[j][1] - P[i][1]) / width)
            if abs(cx) <= K and abs(cy) <= K:
                s, c = acc.get((cx, cy), (0.0, 0))
                acc[(cx, cy)] = (s + 0.5 * (zv[i] - zv[j]) ** 2, c + 1)
    keys = sorted(acc)
    return RichResult(
        payload={
            "dx": [k[0] * width for k in keys],
            "dy": [k[1] * width for k in keys],
            "gamma": [acc[k][0] / acc[k][1] for k in keys],
            "np": [acc[k][1] for k in keys],
        }
    )


def _wls_sills(sv, comps, w):
    """Weighted least squares psills of the components given their ranges (linear in the psills)."""
    h = sv["dist"]
    g = sv["gamma"]
    F = [[(1.0 if c.get("model") == "Nug" else _unit(x, c)) for c in comps] for x in h]
    p = len(comps)
    A = [[ssum(w[t] * F[t][a] * F[t][b] for t in range(len(h))) for b in range(p)] for a in range(p)]
    y = [ssum(w[t] * F[t][a] * g[t] for t in range(len(h))) for a in range(p)]
    Ai = [[float(v) for v in r] for r in inverse(A)]
    s = [ssum(Ai[a][b] * y[b] for b in range(p)) for a in range(p)]
    sse = ssum(w[t] * (g[t] - ssum(s[a] * F[t][a] for a in range(p))) ** 2 for t in range(len(h)))
    return s, sse


def fit_variogram(sample, model, *, method: int = 7, fit_ranges: bool = True) -> RichResult:
    r"""Weighted least squares fit of a (nested) variogram model to a sample variogram.

    Minimises ``sum w_k (gamma_k - gamma(h_k; theta))^2`` with the weights of
    ``gstat::fit.variogram``: ``fit.method`` 1 ``N_k``, 6 unweighted, 7
    ``N_k / h_k^2`` (the default); 2 is Cressie's criterion ``sum N_k
    (gamma_k - gamma(h_k; theta))^2 / gamma(h_k; theta)^2``, minimised
    jointly from the method-7 fit.  Otherwise the partial sills enter
    linearly and are solved exactly for given ranges, the ranges optimised
    on the log scale by Nelder-Mead (Cressie 1985).

    :param sample: Output of :func:`sample_variogram` (or a dict with
        ``np``, ``dist``, ``gamma``).
    :param model: Initial component(s); ranges are starting values.
    :param method: 1, 2, 6 or 7.
    :param fit_ranges: Keep the ranges fixed when false.
    :return: :class:`RichResult` with the fitted ``model`` and ``sse``.

    References
    ----------
    Cressie, N. (1985). Fitting variogram models by weighted least squares.
    *Mathematical Geology*, 17(5), 563-586.

    Examples
    --------
    >>> sv = {"np": [10, 12, 9], "dist": [0.5, 1.0, 1.5], "gamma": [0.4, 0.63, 0.78]}
    >>> f = fit_variogram(sv, {"model": "Exp", "psill": 1.0, "range": 1.0}, fit_ranges=False)
    >>> round(f.model[0]["psill"], 6)
    1.007195
    """
    sv = {k: [float(v) for v in sample[k]] for k in ("np", "dist", "gamma")}
    comps = [dict(c) for c in _comps(model)]
    if method not in (1, 2, 6, 7):
        raise ValueError("method must be 1, 2, 6 or 7")
    nr = [i for i, c in enumerate(comps) if c.get("model") != "Nug"]

    def weights(cs):
        if method == 1:
            return sv["np"]
        if method == 6:
            return [1.0] * len(sv["np"])
        return [n / (h * h) for n, h in zip(sv["np"], sv["dist"])]

    def solve(logr, w):
        cs = [dict(c) for c in comps]
        for i, lr in zip(nr, logr):
            cs[i]["range"] = math.exp(lr)
        s, sse = _wls_sills(sv, cs, w)
        for c, v in zip(cs, s):
            c["psill"] = v
        return cs, sse

    x0 = [math.log(float(comps[i].get("range", 1.0))) for i in nr]
    if method == 2:
        # Cressie (1985): minimise sum N (g - gamma)^2 / gamma^2 jointly, from the method-7 fit
        start = fit_variogram(sv, comps, method=7, fit_ranges=fit_ranges)["model"]
        ps0 = [float(c["psill"]) for c in start]
        lr0 = [math.log(float(start[i]["range"])) for i in nr] if fit_ranges else []

        def build(v):
            cs = [dict(c) for c in start]
            for c, p in zip(cs, v[: len(cs)]):
                c["psill"] = p
            for i, lr in zip(nr, v[len(cs) :]):
                cs[i]["range"] = math.exp(lr)
            return cs

        def crit(v):
            g = vgm_semivariance(sv["dist"], build(v))
            if min(g) <= 0:
                return float("inf")
            return ssum(n * (o - e) ** 2 / (e * e) for n, o, e in zip(sv["np"], sv["gamma"], g))

        x = _nelder_mead(crit, ps0 + lr0)
        cs = build(x)
        return RichResult(payload={"model": cs, "sse": crit(x)})
    w = weights(comps)
    x = list(x0)
    if fit_ranges and x:
        x = _nelder_mead(lambda v: solve(v, w)[1], x)
    cs, sse = solve(x, w)
    return RichResult(payload={"model": cs, "sse": sse})


def _profile(zv, P, X, shape, phi, nu2, reml):
    """Profiled Gaussian log-likelihood at range ``phi`` and relative nugget ``nu2``."""
    from ._mvcore import cholesky, logdet_chol

    n, p = len(zv), len(X[0])
    c = dict(shape, range=phi, psill=1.0)
    R = [
        [(1.0 + nu2 if i == j else 1.0 - _unit(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]), c)) for j in range(n)]
        for i in range(n)
    ]
    L = cholesky(R)
    Ri = [[float(v) for v in r] for r in inverse(R)]
    RiX = [[ssum(Ri[i][k] * X[k][a] for k in range(n)) for a in range(p)] for i in range(n)]
    A = [[ssum(X[i][a] * RiX[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Ai = [[float(v) for v in r] for r in inverse(A)]
    b = [ssum(Ai[a][c2] * ssum(RiX[i][c2] * zv[i] for i in range(n)) for c2 in range(p)) for a in range(p)]
    e = [zv[i] - ssum(X[i][a] * b[a] for a in range(p)) for i in range(n)]
    S = ssum(e[i] * ssum(Ri[i][j] * e[j] for j in range(n)) for i in range(n))
    m = n - p if reml else n
    s2 = S / m
    ll = -0.5 * (m * math.log(2.0 * math.pi) + logdet_chol(L) + m * math.log(s2) + m)
    if reml:
        XtX = [[ssum(X[i][a] * X[i][c2] for i in range(n)) for c2 in range(p)] for a in range(p)]
        ll += 0.5 * (logdet_chol(cholesky(XtX)) - logdet_chol(cholesky(A)))
    return ll, s2, b


def _lik_inputs(z, coords, X):
    zv = [float(v) for v in np.asarray(z, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    Xm = [[1.0]] * len(zv) if X is None else [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    if len(P) != len(zv) or len(Xm) != len(zv):
        raise ValueError("coords and X must match z")
    return zv, P, Xm


def variogram_loglik(z, coords, model: dict, *, nugget: float = 0.0, X=None, method: str = "ML") -> float:
    r"""Gaussian log-likelihood of data under a covariance model, as ``geoR::likfit``'s objective.

    ``Sigma = psill R(range) + nugget I`` with the correlation ``R`` of the
    component ``model`` (:func:`vgm_semivariance` shapes) and the trend
    ``X beta`` at its GLS estimate.  ML: ``-(n log 2 pi + log|Sigma| +
    r' Sigma^{-1} r)/2``; REML adds ``(log|X'X| - log|X' Sigma^{-1} X|) / 2``
    and uses ``n - p`` (Harville 1974, as geoR; Diggle and Ribeiro 2007).

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> round(variogram_loglik([0.5, -0.2, 0.9], [(0, 0), (1, 0), (0, 1)], m, nugget=0.1), 6)
    -3.137442
    """
    if method not in ("ML", "REML"):
        raise ValueError("method must be ML or REML")
    zv, P, Xm = _lik_inputs(z, coords, X)
    from ._mvcore import cholesky, logdet_chol

    n, p = len(zv), len(Xm[0])
    ps = float(model.get("psill", 1.0))
    c = dict(model, psill=1.0)
    S = [
        [
            ps * (1.0 - _unit(math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]), c)) if i != j else ps + nugget
            for j in range(n)
        ]
        for i in range(n)
    ]
    Si = [[float(v) for v in r] for r in inverse(S)]
    SiX = [[ssum(Si[i][k] * Xm[k][a] for k in range(n)) for a in range(p)] for i in range(n)]
    A = [[ssum(Xm[i][a] * SiX[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Ai = [[float(v) for v in r] for r in inverse(A)]
    b = [ssum(Ai[a][c2] * ssum(SiX[i][c2] * zv[i] for i in range(n)) for c2 in range(p)) for a in range(p)]
    e = [zv[i] - ssum(Xm[i][a] * b[a] for a in range(p)) for i in range(n)]
    q = ssum(e[i] * ssum(Si[i][j] * e[j] for j in range(n)) for i in range(n))
    m = n - p if method == "REML" else n
    ll = -0.5 * (m * math.log(2.0 * math.pi) + logdet_chol(cholesky(S)) + q)
    if method == "REML":
        XtX = [[ssum(Xm[i][a] * Xm[i][c2] for i in range(n)) for c2 in range(p)] for a in range(p)]
        ll += 0.5 * (logdet_chol(cholesky(XtX)) - logdet_chol(cholesky(A)))
    return ll


def likfit(
    z, coords, model: dict, *, X=None, method: str = "ML", fix_nugget: bool = False, nugget: float = 0.0
) -> RichResult:
    r"""Maximum likelihood (ML or REML) fit of a Gaussian covariance model, as ``geoR::likfit``.

    Parameters ``psill`` (``sigma^2``), ``range`` (``phi``) and ``nugget``
    (``tau^2``) of one component ``model`` (``kappa`` held fixed) with the
    trend ``X beta``: ``beta`` and ``sigma^2`` are profiled out in closed
    form and ``(log phi, log tau^2/sigma^2)`` maximised by Nelder-Mead from
    ``model``'s range and ``nugget`` (Mardia and Marshall 1984; Diggle and
    Ribeiro 2007, chapter 5).  ``fix_nugget`` holds ``tau^2 = nugget``
    (profiling ``sigma^2`` numerically over ``log phi, log sigma^2``).

    :return: :class:`RichResult` with ``psill``, ``range``, ``nugget``,
        ``beta``, ``loglik``, ``npars``, ``AIC`` and ``BIC``.

    References
    ----------
    Mardia, K. V. and Marshall, R. J. (1984). Maximum likelihood estimation
    of models for residual covariance in spatial regression. *Biometrika*,
    71(1), 135-146.
    Diggle, P. J. and Ribeiro, P. J. (2007). *Model-based Geostatistics*.
    Springer, New York.

    Examples
    --------
    >>> P = [(float(i % 5), float(i // 5)) for i in range(15)]
    >>> z = [0.3, 1.1, 0.8, -0.2, 0.5, 1.4, 0.9, 0.1, -0.4, 0.6, 1.2, 0.7, 0.0, 0.4, 1.0]
    >>> f = likfit(z, P, {"model": "Exp", "range": 1.0}, nugget=0.1, fix_nugget=True)
    >>> round(f.loglik, 4)
    -10.4257
    """
    if method not in ("ML", "REML"):
        raise ValueError("method must be ML or REML")
    zv, P, Xm = _lik_inputs(z, coords, X)
    n, p = len(zv), len(Xm[0])
    reml = method == "REML"
    shape = {k: v for k, v in model.items() if k not in ("psill", "range", "nugget")}
    if not fix_nugget:
        var0 = ssum((v - ssum(zv) / n) ** 2 for v in zv) / n
        x0 = [math.log(float(model.get("range", 1.0))), math.log(max(float(nugget), 1e-3 * var0) / var0)]

        def neg(v):
            try:
                return -_profile(zv, P, Xm, shape, math.exp(v[0]), math.exp(v[1]), reml)[0]
            except (ValueError, ZeroDivisionError, OverflowError):
                return float("inf")

        x = _nelder_mead(neg, x0)
        ll, s2, b = _profile(zv, P, Xm, shape, math.exp(x[0]), math.exp(x[1]), reml)
        out = {"psill": s2, "range": math.exp(x[0]), "nugget": s2 * math.exp(x[1])}
        k = p + 3
    else:
        c0 = dict(shape)

        def negf(v):
            try:
                return -variogram_loglik(
                    zv, P, dict(c0, psill=math.exp(v[1]), range=math.exp(v[0])), nugget=nugget, X=Xm, method=method
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                return float("inf")

        var0 = ssum((v - ssum(zv) / n) ** 2 for v in zv) / n
        x = _nelder_mead(negf, [math.log(float(model.get("range", 1.0))), math.log(var0)])
        ll = -negf(x)
        out = {"psill": math.exp(x[1]), "range": math.exp(x[0]), "nugget": float(nugget)}
        b = _profile(zv, P, Xm, shape, out["range"], out["nugget"] / out["psill"], reml)[2]
        k = p + 2
    out.update({"beta": b, "loglik": ll, "npars": k, "AIC": -2.0 * ll + 2.0 * k, "BIC": -2.0 * ll + k * math.log(n)})
    return RichResult(payload=out)


def select_variogram_model(z, coords, models, *, X=None, method: str = "ML") -> RichResult:
    r"""Choose among candidate covariance models by AIC of their :func:`likfit` fits.

    Examples
    --------
    >>> P = [(float(i % 5), float(i // 5)) for i in range(15)]
    >>> z = [0.3, 1.1, 0.8, -0.2, 0.5, 1.4, 0.9, 0.1, -0.4, 0.6, 1.2, 0.7, 0.0, 0.4, 1.0]
    >>> select_variogram_model(z, P, [{"model": "Exp", "range": 1.0}, {"model": "Gau", "range": 1.0}]).best
    1
    """
    fits = [likfit(z, coords, m, X=X, method=method, nugget=0.1) for m in models]
    aic = [f["AIC"] for f in fits]
    best = min(range(len(fits)), key=lambda i: aic[i])
    return RichResult(payload={"fits": fits, "AIC": aic, "best": best})


def _nelder_mead(f, x0, *, tol: float = 1e-14, maxit: int = 5000):
    n = len(x0)
    S = [list(x0)] + [[x0[j] + (0.1 if j == i else 0.0) for j in range(n)] for i in range(n)]
    F = [f(s) for s in S]
    for _ in range(maxit):
        o = sorted(range(n + 1), key=lambda i: F[i])
        S, F = [S[i] for i in o], [F[i] for i in o]
        if (
            abs(F[-1] - F[0]) <= tol * (abs(F[0]) + 1e-300)
            and max(abs(S[i][j] - S[0][j]) for i in range(1, n + 1) for j in range(n)) < 1e-10
        ):
            break
        c = [ssum(S[i][j] for i in range(n)) / n for j in range(n)]
        xr = [c[j] + (c[j] - S[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < F[0]:
            xe = [c[j] + 2.0 * (c[j] - S[-1][j]) for j in range(n)]
            fe = f(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:
            S[-1], F[-1] = xr, fr
        else:
            xc = [c[j] + 0.5 * (S[-1][j] - c[j]) for j in range(n)]
            fc = f(xc)
            if fc < F[-1]:
                S[-1], F[-1] = xc, fc
            else:
                S = [S[0]] + [[S[0][j] + 0.5 * (S[i][j] - S[0][j]) for j in range(n)] for i in range(1, n + 1)]
                F = [F[0]] + [f(s) for s in S[1:]]
    i = min(range(n + 1), key=lambda k: F[k])
    return S[i]


def cheatsheet() -> str:
    return "vgm_semivariance / sample_variogram / variogram_map / fit_variogram -> gstat-style variography."
