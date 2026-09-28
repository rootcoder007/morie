# morie.fn -- function file (rootcoder007/morie)
"""Variogram extras: n-dimensional directional sample variograms (classical, madogram, rodogram, cross), GSLIB
3-D anisotropy and zonal components, permutation and bootstrap envelopes, jackknife lag standard errors, variogram
cloud box statistics, fractal dimension and Hurst exponent, and windowed temporal semivariance."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .vgmods import vgm_semivariance

__all__ = [
    "sample_variogram_nd",
    "anisotropic_lag",
    "zonal_semivariance",
    "variogram_envelope",
    "variogram_jackknife",
    "variogram_cloud_box",
    "variogram_fractal",
    "windowed_semivariance",
]


def _pts(a):
    return [tuple(float(v) for v in (r if isinstance(r, (list, tuple)) else [r])) for r in a]


def _q7(s, p):
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _bins(P, z, z2, boundaries, direction, tol_deg, estimator):
    nb = len(boundaries) - 1
    acc = [[] for _ in range(nb)]
    dist = [[] for _ in range(nb)]
    u = None
    if direction is not None:
        nrm = math.sqrt(ssum(v * v for v in direction))
        u = [v / nrm for v in direction]
        ct = math.cos(math.radians(tol_deg))
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            h = [b - a for a, b in zip(P[i], P[j])]
            d = math.sqrt(ssum(v * v for v in h))
            if d == 0:
                continue
            if u is not None and abs(ssum(a * b for a, b in zip(h, u))) / d < ct - 1e-12:
                continue
            k = next((k for k in range(nb) if boundaries[k] < d <= boundaries[k + 1]), None)
            if k is None:
                continue
            dz = z[i] - z[j]
            if estimator == "classical":
                v = dz * dz
            elif estimator == "madogram":
                v = abs(dz)
            elif estimator == "rodogram":
                v = math.sqrt(abs(dz))
            elif estimator == "cross":
                v = dz * (z2[i] - z2[j])
            else:
                raise ValueError("estimator must be classical, madogram, rodogram or cross")
            acc[k].append(v)
            dist[k].append(d)
    return acc, dist


def sample_variogram_nd(
    z, coords, boundaries, *, estimator: str = "classical", direction=None, tol: float = 22.5, z2=None
) -> RichResult:
    r"""Sample variogram in any dimension (e.g. 3-D bodies, space-time with scaled time), optionally directional.

    Pairs with ``b_k < d <= b_{k+1}`` (and, with ``direction``, an angle to
    the direction or its opposite within ``tol`` degrees) give per bin the
    number of pairs, mean distance and ``gamma = (1/2N) sum v`` with ``v =
    dz^2`` (classical), ``|dz|`` (madogram), ``|dz|^{1/2}`` (rodogram) or
    ``dz_1 dz_2`` (cross-variogram of ``z`` and ``z2``; Wackernagel 2003).
    Bins with no pairs are dropped. The result can be passed to
    ``morie.fn.vgmods.fit_variogram``.

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB*, 2nd ed., section III.1.
    Wackernagel, H. (2003). *Multivariate Geostatistics*, 3rd ed. Springer.

    Examples
    --------
    >>> r = sample_variogram_nd([1.0, 2.0, 4.0], [(0, 0, 0), (1, 0, 0), (2, 0, 0)], [0.0, 1.5, 2.5])
    >>> r.gamma, r.np
    ([1.25, 4.5], [2, 1])
    """
    P = _pts(coords)
    zv = [float(v) for v in z]
    zz = [float(v) for v in z2] if z2 is not None else None
    B = [float(v) for v in boundaries]
    acc, dist = _bins(P, zv, zz, B, direction, tol, estimator)
    np_, g, dd = [], [], []
    for a, d in zip(acc, dist):
        if a:
            np_.append(len(a))
            g.append(ssum(a) / (2 * len(a)))
            dd.append(ssum(d) / len(d))
    return RichResult(payload={"np": np_, "dist": dd, "gamma": g, "estimator": estimator})


def anisotropic_lag(
    h, *, azimuth: float = 0.0, dip: float = 0.0, rake: float = 0.0, ratio1: float = 1.0, ratio2: float = 1.0
) -> float:
    r"""Isotropic-equivalent lag of a 3-D separation vector under GSLIB geometric anisotropy.

    Rotation by ``azimuth`` (clockwise from north about z), ``dip`` (down
    from horizontal) and ``rake`` (about the major axis), then the second and
    third axes divided by ``ratio1``, ``ratio2`` (minor/major range ratios)
    (Deutsch and Journel 1998, section II.3, ``setrot``); evaluate an
    isotropic model at the result to obtain the anisotropic variogram.

    Examples
    --------
    >>> round(anisotropic_lag((0.0, 1.0, 0.0)), 6), round(anisotropic_lag((1.0, 0.0, 0.0), ratio1=0.5), 6)
    (1.0, 2.0)
    """
    x, y, zc = (float(v) for v in h)
    a = math.radians(90.0 - azimuth)
    b = math.radians(-dip)
    t = math.radians(rake)
    ca, sa, cb, sb, ct, st = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(t), math.sin(t)
    rot = [
        [cb * ca, cb * sa, -sb],
        [(-ct * sa + st * sb * ca) / ratio1, (ct * ca + st * sb * sa) / ratio1, (st * cb) / ratio1],
        [(st * sa + ct * sb * ca) / ratio2, (-st * ca + ct * sb * sa) / ratio2, (ct * cb) / ratio2],
    ]
    v = [r[0] * x + r[1] * y + r[2] * zc for r in rot]
    return math.sqrt(ssum(c * c for c in v))


def zonal_semivariance(h, model, zonal_model, *, axis: int = 2) -> float:
    r"""Nested variogram with a zonal anisotropy: ``gamma(h) = gamma_1(|h|) + gamma_2(|h_axis|)``.

    The zonal structure varies only along coordinate ``axis`` (e.g. vertical
    stratification in 3-D; Journel and Huijbregts 1978); models are
    ``vgmods`` dicts.

    References
    ----------
    Journel, A. G. and Huijbregts, C. J. (1978). *Mining Geostatistics*.
    Academic Press.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> round(zonal_semivariance((0.0, 0.0, 1.0), m, m), 6)
    1.264241
    """
    hv = [float(v) for v in h]
    d = math.sqrt(ssum(v * v for v in hv))
    return vgm_semivariance(d, model) + vgm_semivariance(abs(hv[axis]), zonal_model)


def variogram_envelope(
    z, coords, boundaries, *, nsim: int = 99, level: float = 0.95, method: str = "permutation", seed: int = 1
) -> RichResult:
    r"""Monte Carlo envelope for the classical sample variogram under spatial independence (``geoR::variog.mc.env``).

    ``permutation`` reassigns the data values to the locations at random;
    ``bootstrap`` resamples them with replacement (Philox stream ``s``);
    pointwise ``(1 - level)/2`` and ``(1 + level)/2`` quantiles (type 7) of the
    simulated ``gamma`` per bin give the envelope; observed bins outside it
    suggest spatial structure (Diggle and Ribeiro 2007, section 5.2).

    References
    ----------
    Diggle, P. J. and Ribeiro, P. J. (2007). *Model-based Geostatistics*.
    Springer.

    Examples
    --------
    >>> r = variogram_envelope([1.0, 2.0, 3.0, 4.0], [(0,), (1,), (2,), (3,)], [0.0, 1.5, 3.5], nsim=19)
    >>> len(r.lower), all(lo <= hi for lo, hi in zip(r.lower, r.upper))
    (2, True)
    """
    P = _pts(coords)
    zv = [float(v) for v in z]
    obs = sample_variogram_nd(zv, P, boundaries)
    n = len(zv)
    sims = []
    for s in range(nsim):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        if method == "permutation":
            idx = list(range(n))
            for t in range(n):
                j = t + int(u[t] * (n - t))
                idx[t], idx[j] = idx[j], idx[t]
            zs = [zv[i] for i in idx]
        elif method == "bootstrap":
            zs = [zv[min(n - 1, int(v * n))] for v in u]
        else:
            raise ValueError("method must be permutation or bootstrap")
        sims.append(sample_variogram_nd(zs, P, boundaries)["gamma"])
    lo_p, hi_p = (1 - level) / 2, (1 + level) / 2
    k = len(obs["gamma"])
    lower = [_q7(sorted(r[b] for r in sims), lo_p) for b in range(k)]
    upper = [_q7(sorted(r[b] for r in sims), hi_p) for b in range(k)]
    out = [not (lo <= g <= hi) for g, lo, hi in zip(obs["gamma"], lower, upper)]
    return RichResult(
        payload={"gamma": obs["gamma"], "dist": obs["dist"], "lower": lower, "upper": upper, "outside": out}
    )


def variogram_jackknife(z, coords, boundaries) -> RichResult:
    r"""Delete-one-datum jackknife standard errors of the classical sample variogram per lag bin (Shafer and Varljen 1990).

    ``gamma_(-i)`` is recomputed without datum ``i``; the jackknife standard
    error is ``sqrt((n - 1)/n sum (gamma_(-i) - mean)^2)`` and the
    bias-corrected estimate ``n gamma - (n - 1) mean``. Bins must keep pairs
    after every deletion.

    References
    ----------
    Shafer, J. M. and Varljen, M. D. (1990). Approximation of confidence
    limits on sample semivariograms from single realizations of spatially
    correlated random fields. *Water Resources Research*, 26(8), 1787-1802.

    Examples
    --------
    >>> r = variogram_jackknife([1.0, 2.0, 4.0, 3.0], [(0,), (1,), (2,), (3,)], [0.0, 1.5])
    >>> round(r.gamma[0], 6), round(r.se[0], 6)
    (1.0, 0.649519)
    """
    P = _pts(coords)
    zv = [float(v) for v in z]
    full = sample_variogram_nd(zv, P, boundaries)
    n = len(zv)
    reps = [sample_variogram_nd(zv[:i] + zv[i + 1 :], P[:i] + P[i + 1 :], boundaries)["gamma"] for i in range(n)]
    k = len(full["gamma"])
    if any(len(r) != k for r in reps):
        raise ValueError("a lag bin loses all its pairs when a datum is removed")
    mean = [ssum(r[b] for r in reps) / n for b in range(k)]
    se = [math.sqrt((n - 1) / n * ssum((r[b] - mean[b]) ** 2 for r in reps)) for b in range(k)]
    bc = [n * g - (n - 1) * m for g, m in zip(full["gamma"], mean)]
    return RichResult(payload={"gamma": full["gamma"], "se": se, "bias_corrected": bc, "dist": full["dist"]})


def variogram_cloud_box(z, coords, boundaries) -> RichResult:
    r"""Box-plot statistics of the variogram cloud ``(z_i - z_j)^2 / 2`` per lag bin (Cressie 1993, section 2.2.2).

    Per bin: minimum, lower quartile, median, upper quartile, maximum (type
    7 quantiles), whiskers at the most extreme values within 1.5 IQR of the
    quartiles, and the number of outliers beyond them.

    Examples
    --------
    >>> r = variogram_cloud_box([1.0, 2.0, 4.0, 3.0], [(0,), (1,), (2,), (3,)], [0.0, 1.5])
    >>> r.median, r.n
    ([0.5], [3])
    """
    P = _pts(coords)
    zv = [float(v) for v in z]
    acc, _ = _bins(P, zv, None, [float(v) for v in boundaries], None, 90.0, "classical")
    out = {k: [] for k in ("min", "q1", "median", "q3", "max", "lower_whisker", "upper_whisker", "outliers", "n")}
    for a in acc:
        if not a:
            continue
        s = sorted(v / 2 for v in a)
        q1, q3 = _q7(s, 0.25), _q7(s, 0.75)
        iqr = q3 - q1
        inside = [v for v in s if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
        for key, val in (
            ("min", s[0]),
            ("q1", q1),
            ("median", _q7(s, 0.5)),
            ("q3", q3),
            ("max", s[-1]),
            ("lower_whisker", inside[0]),
            ("upper_whisker", inside[-1]),
            ("outliers", len(s) - len(inside)),
            ("n", len(s)),
        ):
            out[key].append(val)
    return RichResult(payload=out)


def variogram_fractal(dist, gamma, *, dim: int = 1, max_lag: float | None = None) -> RichResult:
    r"""Hurst exponent and fractal dimension from the small-lag power law ``gamma(h) ~ c h^{2H}``.

    OLS of ``log gamma`` on ``log h`` over lags up to ``max_lag`` gives slope
    ``2H``; the fractal dimension of a ``dim``-dimensional self-affine surface
    is ``D = dim + 1 - H`` (Mandelbrot 1983; Burrough 1981).

    References
    ----------
    Burrough, P. A. (1981). Fractal dimensions of landscapes and other
    environmental data. *Nature*, 294, 240-242.

    Examples
    --------
    >>> r = variogram_fractal([1.0, 2.0, 4.0], [1.0, 2.0, 4.0])
    >>> r.hurst, r.fractal_dimension
    (0.5, 1.5)
    """
    pts = [
        (math.log(float(h)), math.log(float(g)))
        for h, g in zip(dist, gamma)
        if g > 0 and h > 0 and (max_lag is None or h <= max_lag)
    ]
    n = len(pts)
    mx, my = ssum(p[0] for p in pts) / n, ssum(p[1] for p in pts) / n
    slope = ssum((p[0] - mx) * (p[1] - my) for p in pts) / ssum((p[0] - mx) ** 2 for p in pts)
    H = slope / 2
    return RichResult(
        payload={"hurst": H, "fractal_dimension": dim + 1 - H, "slope": slope, "intercept": my - slope * mx}
    )


def windowed_semivariance(series, lags, *, window: int, step: int = 1) -> RichResult:
    r"""Temporal semivariance ``gamma(k) = sum (x_{t+k} - x_t)^2 / (2 N_k)`` within moving windows of ``window`` points.

    Windows start at ``0, step, 2 step, ...``; returns the window starts and
    one semivariance row per window (tracking nonstationary variability).

    Examples
    --------
    >>> r = windowed_semivariance([0.0, 1.0, 0.0, 1.0, 5.0, 0.0], [1], window=4, step=2)
    >>> r.gamma
    [[0.5], [7.0]]
    """
    x = [float(v) for v in series]
    L = [int(v) for v in lags]
    starts, rows = [], []
    for s in range(0, len(x) - window + 1, step):
        w = x[s : s + window]
        row = []
        for k in L:
            d = [w[t + k] - w[t] for t in range(len(w) - k)]
            row.append(ssum(v * v for v in d) / (2 * len(d)) if d else float("nan"))
        starts.append(s)
        rows.append(row)
    return RichResult(payload={"start": starts, "gamma": rows, "lags": L})


def cheatsheet() -> str:
    return "sample_variogram_nd / anisotropic_lag / variogram_envelope / variogram_jackknife / variogram_fractal."
