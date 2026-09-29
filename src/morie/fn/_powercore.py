# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels of the power-analysis front ends (``pwr_t``, ``pwr_p``, ``pwr_av``, ``i_pwr``).

The power functions of R's ``power.t.test``, ``power.prop.test`` and the
noncentral-F power of the one-way ANOVA, with a bracketed bisection that
solves for the missing quantity to ``1e-12`` (R's ``uniroot`` default,
``eps^0.25``, stops near ``1e-4``). R twin: ``R/PowerAnalysis.R``
(helpers ``.pwa_*``).
"""

from __future__ import annotations

import math

from ._stats_core import f as _f
from ._stats_core import ncf, nct
from ._stats_core import norm as _norm
from ._stats_core import t as _t

__all__: list = []


def bisect(fn, lo, hi, tol=1e-12, max_iter=300):
    """Root of ``fn`` on ``[lo, hi]`` (sign change required) by bisection to ``tol`` relative."""
    flo, fhi = fn(lo), fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0) == (fhi > 0):
        raise ValueError(f"no solution in [{lo:g}, {hi:g}]")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo <= tol * max(1.0, abs(mid)):
            break
        fm = fn(mid)
        if fm == 0.0:
            return mid
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_up(fn, lo, hi_max):
    """Root of the increasing ``fn`` above ``lo``: double an upper bracket from ``2 lo`` up to ``hi_max``, then bisect."""
    hi = 2.0 * lo
    while fn(hi) < 0.0 and hi < hi_max:
        lo, hi = hi, min(2.0 * hi, hi_max)
    return bisect(fn, lo, hi)


def t_power(n, delta, sd, alpha, tsample, tside, strict):
    """``power.t.test``: ``P(T' > t_{1 - alpha/tside, nu})`` (+ lower tail when strict), ``ncp = sqrt(n/tsample) delta/sd``."""
    nu = max(1e-7, n - 1.0) * tsample
    qu = float(_t.ppf(1.0 - alpha / tside, nu))
    ncp = math.sqrt(n / tsample) * delta / sd
    p = float(nct.sf(qu, nu, ncp))
    if strict and tside == 2:
        p += float(nct.cdf(-qu, nu, ncp))
    return p


def prop_power(n, p1, p2, alpha, tside, strict):
    """``power.prop.test`` (Fleiss 1981 normal approximation with pooled null variance)."""
    qu = float(_norm.ppf(1.0 - alpha / tside))
    d = abs(p1 - p2)
    v1, v2 = p1 * (1 - p1), p2 * (1 - p2)
    pbar = (p1 + p2) / 2
    s0 = math.sqrt(2 * pbar * (1 - pbar))
    s1 = math.sqrt(v1 + v2)
    p = float(_norm.cdf((math.sqrt(n) * d - qu * s0) / s1))
    if strict and tside == 2:
        p += float(_norm.sf((math.sqrt(n) * d + qu * s0) / s1))
    return p


def cohen_h_power(n, p1, p2, alpha, tside):
    """Arcsine (Cohen's h) normal approximation, as ``pwr::pwr.2p.test`` (both tails when two-sided)."""
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    delta = h * math.sqrt(n / 2.0)
    qu = float(_norm.ppf(1.0 - alpha / tside))
    p = float(_norm.sf(qu - delta))
    if tside == 2:
        p += float(_norm.cdf(-qu - delta))
    return p


def anova_power(ntot, k, f, alpha, df1=None):
    """Noncentral-F power, ``ncp = f^2 N``, ``df1 = k - 1`` (or given), ``df2 = N - df1 - 1`` for ``df1`` given else ``N - k``."""
    d1 = float(k - 1) if df1 is None else float(df1)
    d2 = ntot - float(k) if df1 is None else ntot - d1 - 1.0
    if d1 <= 0 or d2 <= 0:
        raise ValueError("degrees of freedom must be positive")
    crit = float(_f.ppf(1.0 - alpha, d1, d2))
    return float(ncf.sf(crit, d1, d2, f * f * ntot))
