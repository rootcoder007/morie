# morie.fn -- function file (rootcoder007/morie)
"""Climate indices and trend-seasonal decomposition: the station-based North Atlantic Oscillation
index (normalised sea-level-pressure difference) and the NOAA/ESRL curve fit of CO2 records
(polynomial trend plus annual harmonics, Thoning et al. 1989)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["nao_station_index", "co2_curve_fit"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _standardise(v, groups, base):
    out = [0.0] * len(v)
    for g in sorted(set(groups)):
        idx = [i for i in range(len(v)) if groups[i] == g]
        ref = [v[i] for i in idx if base[i]]
        m = ssum(ref) / len(ref)
        s = math.sqrt(ssum((r - m) ** 2 for r in ref) / (len(ref) - 1))
        for i in idx:
            out[i] = (v[i] - m) / s
    return out


def nao_station_index(slp_south, slp_north, months=None, base=None):
    r"""Station-based NAO index: normalised SLP at the southern station minus that at the northern one.

    Each series is standardised by its mean and standard deviation over the
    base period (``base``: boolean mask, default all), separately for each
    calendar month when ``months`` is given (Hurrell 1995: Lisbon or Ponta
    Delgada minus Stykkisholmur/Reykjavik).

    References
    ----------
    Hurrell, J. W. (1995). Decadal trends in the North Atlantic Oscillation:
    regional temperatures and precipitation. *Science* 269, 676-679.

    Examples
    --------
    >>> [round(v, 12) for v in nao_station_index([1020.0, 1024.0, 1018.0], [1000.0, 996.0, 1004.0])]
    [-0.218217890236, 2.09108945118, -1.872871560944]
    """
    s, n = _vec(slp_south), _vec(slp_north)
    g = [0] * len(s) if months is None else [int(v) for v in months]
    b = [True] * len(s) if base is None else [bool(v) for v in base]
    zs, zn = _standardise(s, g, b), _standardise(n, g, b)
    return [a - c for a, c in zip(zs, zn)]


def co2_curve_fit(t, co2, n_poly=3, n_harm=4):
    r"""Polynomial-plus-harmonics fit of an atmospheric CO2 record (NOAA/ESRL CCGCRV function).

    ``f(t) = sum_{j<n_poly} a_j t^j + sum_{k=1}^{n_harm} (b_k sin 2 pi k t + c_k
    cos 2 pi k t)`` by least squares, ``t`` in decimal years (centred at its
    mean for conditioning). Returns the coefficients, the long-term trend (the
    polynomial), the mean seasonal cycle (the harmonics), the residuals and
    the trend growth rate ``df/dt`` (ppm per year).

    References
    ----------
    Thoning, K. W., Tans, P. P. and Komhyr, W. D. (1989). Atmospheric carbon
    dioxide at Mauna Loa Observatory 2. Analysis of the NOAA GMCC data,
    1974-1985. *Journal of Geophysical Research* 94, 8549-8565.

    Examples
    --------
    >>> t = [2000 + i / 12 for i in range(24)]
    >>> y = [370 + 2 * (v - 2000) + 3 * math.sin(2 * math.pi * v) for v in t]
    >>> r = co2_curve_fit(t, y, n_poly=2, n_harm=1)
    >>> [round(v, 9) for v in r.growth_rate[:2]]
    [2.0, 2.0]
    """
    tv, y = _vec(t), _vec(co2)
    t0 = ssum(tv) / len(tv)
    rows = []
    for v in tv:
        u = v - t0
        r = [u**j for j in range(n_poly)]
        for k in range(1, n_harm + 1):
            r += [math.sin(2 * math.pi * k * v), math.cos(2 * math.pi * k * v)]
        rows.append(r)
    p = len(rows[0])
    beta = solve(
        [[ssum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)],
        [ssum(r[a] * v for r, v in zip(rows, y)) for a in range(p)],
    )
    trend = [ssum(r[j] * beta[j] for j in range(n_poly)) for r in rows]
    seas = [ssum(r[j] * beta[j] for j in range(n_poly, p)) for r in rows]
    growth = [ssum(j * beta[j] * (v - t0) ** (j - 1) for j in range(1, n_poly)) for v in tv]
    resid = [a - b - c for a, b, c in zip(y, trend, seas)]
    return RichResult(
        payload={
            "coefficients": beta,
            "t0": t0,
            "trend": trend,
            "seasonal": seas,
            "residuals": resid,
            "growth_rate": growth,
        }
    )


def cheatsheet() -> str:
    return "nao_station_index / co2_curve_fit -> climate indices."
