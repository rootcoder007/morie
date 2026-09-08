# morie.fn -- function file (rootcoder007/morie)
"""McCrary density-discontinuity test for manipulation of a running variable."""

from __future__ import annotations

import math

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["causal_rdd_manipulation"]


def _wls(xs, ys, ws):
    """Weighted least squares of ``y`` on ``(1, x)``; returns the intercept."""
    X = [[math.sqrt(w), math.sqrt(w) * xi] for xi, w in zip(xs, ws)]
    yy = [math.sqrt(w) * yi for yi, w in zip(ys, ws)]
    beta, _, _, _ = T.olsfit(X, yy)
    return beta[0]


def _poly4(mp, val):
    """Degree-4 polynomial fit; returns coefficients and the residual MSE."""
    X = [[1.0, m, m ** 2, m ** 3, m ** 4] for m in mp]
    beta, fitted, resid, _ = T.olsfit(X, val)
    dof = len(mp) - 5
    if dof <= 0:
        raise ValueError("too few bins on one side to fit the degree-4 pilot")
    return beta, sum(r * r for r in resid) / dof


def causal_rdd_manipulation(x, cutoff=0.0, bw=None, binsize=None):
    """McCrary's test for a discontinuity in the density at the cutoff.

    The running variable is binned, the bin heights are turned into a
    density, and a local linear regression with triangular kernel is run
    separately on each side and evaluated at the cutoff.  The statistic
    is the log difference in heights

        ``theta = log fhat_+(c) - log fhat_-(c)``

    with

        ``se(theta) = sqrt( (1/(n h)) (24/5) (1/fhat_+ + 1/fhat_-) )``

    and ``z = theta / se(theta)`` referred to the standard normal.  The
    ``24/5`` is the triangular-kernel constant for a boundary local
    linear estimator; it is not interchangeable with the interior
    constant.

    Defaults follow McCrary's own code: bin width ``2 s n^{-1/2}``, and
    a bandwidth chosen per side from a degree-4 pilot polynomial,

        ``h = 3.348 (mse4 * range / sum f''^2)^{1/5}``

    averaged over the two sides.  The binned densities are zero-padded
    out to one bandwidth on each side so the boundary regressions see
    the empty tail rather than running off the end of the data.

    A rejection here says the density jumps, which is evidence of
    sorting around the threshold; failing to reject is not evidence that
    assignment is as good as random.

    Parameters
    ----------
    x : array-like
        Running variable.
    cutoff : float
        Threshold, which must lie strictly inside the range of ``x``.
    bw : float, optional
        Bandwidth; McCrary's automatic rule if omitted.
    binsize : float, optional
        Bin width; ``2 s n^{-1/2}`` if omitted.

    Returns
    -------
    RichResult
        ``estimate`` (theta), ``se``, ``statistic`` (z), ``p_value``,
        ``fhat_left``, ``fhat_right``, ``bw``, ``binsize``, ``n``,
        ``method``.

    References
    ----------
    McCrary (2008), Manipulation of the running variable in the
    regression discontinuity design: a density test, Journal of
    Econometrics 142:698-714.  Paywalled at Elsevier; the coded form was
    read from McCrary's own implementation as distributed in Drew
    Dimmery's ``rdd`` package, R/DCdensity.R (fetched from the CRAN
    GitHub mirror), which gives the binning, the 3.348 pilot bandwidth
    rule, the zero padding, the triangular weights and
    ``sethetahat <- sqrt((1/(rn*bw)) * (24/5) * ((1/fhatr)+(1/fhatl)))``
    verbatim.
    """
    x = T.vec(x)
    rn = len(x)
    if rn < 20:
        raise ValueError("need at least 20 observations")
    cutoff = float(cutoff)
    mean = sum(x) / rn
    rsd = math.sqrt(sum((xi - mean) ** 2 for xi in x) / (rn - 1))
    rmin, rmax = min(x), max(x)
    if cutoff <= rmin or cutoff >= rmax:
        raise ValueError("cutoff must lie strictly within the range of x")
    b = float(binsize) if binsize is not None else 2.0 * rsd * rn ** -0.5
    lo = math.floor((rmin - cutoff) / b) * b + b / 2.0 + cutoff
    hi = math.floor((rmax - cutoff) / b) * b + b / 2.0 + cutoff
    j = int(math.floor((rmax - rmin) / b)) + 2
    cellval = [0.0] * j
    for xi in x:
        mid = math.floor((xi - cutoff) / b) * b + b / 2.0 + cutoff
        idx = int(round((mid - lo) / b))
        if idx < 0:
            idx = 0
        if idx >= j:
            idx = j - 1
        cellval[idx] += 1.0
    cellval = [c / rn / b for c in cellval]
    cellmp = []
    for i in range(1, j + 1):
        v = lo + (i - 1) * b
        cellmp.append(math.floor((v - cutoff) / b) * b + b / 2.0 + cutoff)
    if bw is None:
        mpl = [m for m in cellmp if m < cutoff]
        mpr = [m for m in cellmp if m >= cutoff]
        vl = [cellval[i] for i in range(j) if cellmp[i] < cutoff]
        vr = [cellval[i] for i in range(j) if cellmp[i] >= cutoff]
        lc, msel = _poly4(mpl, vl)
        rc, mser = _poly4(mpr, vr)
        fppl = [2 * lc[2] + 6 * lc[3] * m + 12 * lc[4] * m * m for m in mpl]
        fppr = [2 * rc[2] + 6 * rc[3] * m + 12 * rc[4] * m * m for m in mpr]
        hleft = 3.348 * (msel * (cutoff - lo) / sum(v * v for v in fppl)) ** 0.2
        hright = 3.348 * (mser * (hi - cutoff) / sum(v * v for v in fppr)) ** 0.2
        bw = 0.5 * (hleft + hright)
    bw = float(bw)
    if not any(cutoff - bw < xi < cutoff for xi in x) or not any(cutoff <= xi < cutoff + bw for xi in x):
        raise ValueError("insufficient data within the bandwidth")
    pad = int(math.ceil(bw / b))
    cmp_ = [lo - (pad - i) * b for i in range(pad)] + cellmp + [hi + (i + 1) * b for i in range(pad)]
    cval = [0.0] * pad + cellval + [0.0] * pad
    jp = j + 2 * pad
    dist = [m - cutoff for m in cmp_]
    fhat = {}
    for side in ("left", "right"):
        w = []
        for i in range(jp):
            wi = 1.0 - abs(dist[i] / bw)
            keep = (cmp_[i] < cutoff) if side == "left" else (cmp_[i] >= cutoff)
            w.append(wi * keep if wi > 0 else 0.0)
        sw = sum(w)
        w = [wi / sw * jp for wi in w]
        fhat[side] = _wls(dist, cval, w)
    fl, fr = fhat["left"], fhat["right"]
    if fl <= 0 or fr <= 0:
        raise ValueError("non-positive density estimate at the cutoff")
    theta = math.log(fr) - math.log(fl)
    se = math.sqrt((1.0 / (rn * bw)) * (24.0 / 5.0) * (1.0 / fr + 1.0 / fl))
    z = theta / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return RichResult(
        payload={
            "estimate": float(theta),
            "se": float(se),
            "statistic": float(z),
            "p_value": float(p),
            "fhat_left": float(fl),
            "fhat_right": float(fr),
            "bw": float(bw),
            "binsize": float(b),
            "n": int(rn),
            "method": "McCrary density discontinuity test",
        }
    )


def cheatsheet():
    return "causal_rdd_manipulation(x, cutoff, bw): McCrary log-density jump at the cutoff."


# compact alias per ledger/NAMING.md
rddmanip = causal_rdd_manipulation
