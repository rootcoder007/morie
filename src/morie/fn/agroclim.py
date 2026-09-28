# morie.fn -- function file (rootcoder007/morie)
"""Agroclimatic indicators: growing degree days, chill hours and Utah chill units, LAI from SAVI, rainfall
adequacy, ETCCDI growing season length and cold spell duration."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "growing_degree_days",
    "chill_accumulation",
    "lai_from_savi",
    "rainfall_adequacy",
    "growing_season_length",
    "cold_spell_duration",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def growing_degree_days(tmax, tmin, base: float = 10.0, upper: float | None = None) -> RichResult:
    r"""Growing degree days by the averaging method: ``GDD = max(0, (Tmax + Tmin)/2 - Tbase)`` per day.

    With an ``upper`` threshold the horizontal cutoff applies first: ``Tmax``
    is capped at ``upper`` and ``Tmin`` raised to ``base`` (McMaster and
    Wilhelm 1997, method 2). Returns daily values and their cumulative sum.

    References
    ----------
    McMaster, G. S. and Wilhelm, W. W. (1997). Growing degree-days: one
    equation, two interpretations. *Agricultural and Forest Meteorology*,
    87(4), 291-300.

    Examples
    --------
    >>> r = growing_degree_days([25.0, 32.0, 12.0], [11.0, 20.0, 4.0], base=10.0, upper=30.0)
    >>> r.daily, r.total
    ([8.0, 15.0, 1.0], 24.0)
    """
    hi, lo = _vec(tmax), _vec(tmin)
    d = []
    for a, b in zip(hi, lo):
        if upper is not None:
            a, b = min(a, upper), max(b, base)
        d.append(max(0.0, (a + b) / 2 - base))
    return RichResult(payload={"daily": d, "cumulative": [ssum(d[: i + 1]) for i in range(len(d))], "total": ssum(d)})


def chill_accumulation(hourly_temp) -> RichResult:
    r"""Chill hours (0 to 7.2 C, Weinberger 1950) and Utah chill units (Richardson et al. 1974) from hourly temperatures.

    Utah weights: <= 1.4 C: 0; 1.5-2.4: 0.5; 2.5-9.1: 1; 9.2-12.4: 0.5;
    12.5-15.9: 0; 16.0-18.0: -0.5; > 18: -1 (the bands are applied as
    half-open intervals ``(lower, upper]`` at the stated one-decimal limits).

    References
    ----------
    Richardson, E. A., Seeley, S. D. and Walker, D. R. (1974). A model for
    estimating the completion of rest for 'Redhaven' and 'Elberta' peach
    trees. *HortScience*, 9(4), 331-332.
    Weinberger, J. H. (1950). Chilling requirements of peach varieties.
    *Proceedings of the American Society for Horticultural Science*, 56, 122-128.

    Examples
    --------
    >>> r = chill_accumulation([-1.0, 2.0, 5.0, 8.0, 10.0, 14.0, 17.0, 20.0])
    >>> r.chill_hours, r.utah_units
    (2, 1.5)
    """
    T = _vec(hourly_temp)
    ch = sum(1 for t in T if 0 <= t <= 7.2)

    def w(t):
        if t <= 1.4:
            return 0.0
        if t <= 2.4:
            return 0.5
        if t <= 9.1:
            return 1.0
        if t <= 12.4:
            return 0.5
        if t <= 15.9:
            return 0.0
        if t <= 18.0:
            return -0.5
        return -1.0

    return RichResult(payload={"chill_hours": ch, "utah_units": ssum(w(t) for t in T)})


def lai_from_savi(savi, *, cap: float = 6.0):
    r"""Leaf area index from SAVI (METRIC/SEBAL, Bastiaanssen 1998): ``LAI = -ln((0.69 - SAVI)/0.59)/0.91``.

    Values are 0 for ``SAVI <= 0.1`` and ``cap`` for ``SAVI >= 0.687`` (the
    expression's asymptote at 0.69).

    References
    ----------
    Allen, R. G., Tasumi, M. and Trezza, R. (2007). Satellite-based energy
    balance for mapping evapotranspiration with internalized calibration
    (METRIC) -- model. *Journal of Irrigation and Drainage Engineering*,
    133(4), 380-394.

    Examples
    --------
    >>> [round(v, 6) for v in lai_from_savi([0.05, 0.4, 0.7])]
    [0.0, 0.780485, 6.0]
    """
    out = []
    for s in _vec(savi):
        if s <= 0.1:
            out.append(0.0)
        elif s >= 0.687:
            out.append(cap)
        else:
            out.append(min(cap, -math.log((0.69 - s) / 0.59) / 0.91))
    return out


def rainfall_adequacy(precip, et0, kc=1.0) -> RichResult:
    r"""Rainfall adequacy of a crop: ``ratio = sum P / sum ETc`` and deficit ``max(0, ETc - P)`` per period, ``ETc = Kc ET0``.

    References
    ----------
    Allen, R. G., Pereira, L. S., Raes, D. and Smith, M. (1998). *Crop
    Evapotranspiration*. FAO Irrigation and Drainage Paper 56.

    Examples
    --------
    >>> r = rainfall_adequacy([30.0, 10.0], [40.0, 50.0], kc=[0.5, 1.0])
    >>> r.ratio, r.deficit
    (0.5714285714285714, [0.0, 40.0])
    """
    P, E = _vec(precip), _vec(et0)
    K = [float(kc)] * len(E) if isinstance(kc, (int, float)) else _vec(kc)
    etc = [k * e for k, e in zip(K, E)]
    return RichResult(
        payload={"etc": etc, "ratio": ssum(P) / ssum(etc), "deficit": [max(0.0, a - b) for a, b in zip(etc, P)]}
    )


def _first_run(flags, length, start=0):
    run = 0
    for i in range(start, len(flags)):
        run = run + 1 if flags[i] else 0
        if run == length:
            return i - length + 1
    return None


def growing_season_length(tmean, *, threshold: float = 5.0, span: int = 6, midyear: int = 181) -> int:
    r"""ETCCDI growing season length (GSL) of one year of daily mean temperatures.

    Days from the first run of ``span`` consecutive days with ``T > 5`` C to
    the first run of ``span`` days with ``T < 5`` C starting after
    ``midyear`` (1 July in the Northern Hemisphere: day index 181 of a
    non-leap year, 0-based); the season ends the day before that run and
    reaches year end when no such run occurs. 0 when no season starts.

    References
    ----------
    Zhang, X. et al. (2011). Indices for monitoring changes in extremes
    based on daily temperature and precipitation data. *WIREs Climate
    Change*, 2(6), 851-870.

    Examples
    --------
    >>> growing_season_length([0.0] * 100 + [10.0] * 150 + [0.0] * 115)
    150
    """
    T = _vec(tmean)
    s = _first_run([t > threshold for t in T], span)
    if s is None:
        return 0
    e = _first_run([t < threshold for t in T], span, max(midyear, s + 1))
    return (len(T) if e is None else e) - s


def cold_spell_duration(tmin, threshold, *, span: int = 6) -> RichResult:
    r"""ETCCDI cold spell duration index (CSDI): days in runs of at least ``span`` consecutive days with ``Tmin < threshold``.

    ``threshold`` is the calendar-day 10th percentile of the base period
    (a scalar or one value per day); also returned: the number of spells.

    Examples
    --------
    >>> r = cold_spell_duration([1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 9], 5.0)
    >>> r.csdi, r.spells
    (7, 1)
    """
    T = _vec(tmin)
    th = [float(threshold)] * len(T) if isinstance(threshold, (int, float)) else _vec(threshold)
    days = spells = run = 0
    for t, h in zip(T + [math.inf], th + [-math.inf]):
        if t < h:
            run += 1
        else:
            if run >= span:
                days += run
                spells += 1
            run = 0
    return RichResult(payload={"csdi": days, "spells": spells})


def cheatsheet() -> str:
    return "growing_degree_days / chill_accumulation / lai_from_savi / growing_season_length -> agroclimate."
