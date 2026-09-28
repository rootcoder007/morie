# morie.fn -- function file (rootcoder007/morie)
"""Environmental noise: level statistics, day-evening-night indicators, ISO 9613 propagation, CRTN road noise,
construction and vibration, underwater spreading, annoyance and health dose-response, noise maps and zones."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "equivalent_level",
    "level_statistics",
    "sound_exposure_level",
    "combine_levels",
    "day_evening_night",
    "aircraft_dnl",
    "atmospheric_absorption",
    "point_source_level",
    "attenuation_distance",
    "ground_attenuation",
    "barrier_attenuation",
    "foliage_attenuation",
    "crtn_road_noise",
    "construction_noise",
    "vibration_propagation",
    "underwater_propagation",
    "noise_annoyance",
    "noise_health_risk",
    "noise_map",
    "noise_zones",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _db(e):
    return 10.0 * math.log10(e)


def equivalent_level(levels, durations=None) -> float:
    r"""Equivalent continuous level ``Leq = 10 log10(sum t_i 10^{L_i/10} / sum t_i)`` (equal durations by default).

    Examples
    --------
    >>> round(equivalent_level([60.0, 70.0]), 6)
    67.403627
    """
    L = _vec(levels)
    t = [1.0] * len(L) if durations is None else _vec(durations)
    return _db(ssum(ti * 10 ** (li / 10) for li, ti in zip(L, t)) / ssum(t))


def _q7(x, p):
    s = sorted(x)
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def level_statistics(levels, percentiles=(10, 50, 90)) -> RichResult:
    r"""Statistical levels of a sampled level history: ``L_N`` (exceeded ``N`` percent of the time), Leq, Lmax, Lmin.

    ``L_N`` is the ``1 - N/100`` sample quantile (linear interpolation, R's
    type 7) of equally spaced samples; ``L10``, ``L50`` and ``L90`` are the
    usual intrusive, median and background levels.

    Examples
    --------
    >>> s = level_statistics([50.0, 52.0, 55.0, 60.0, 71.0])
    >>> s.L10, s.L50, s.L90, s.Lmax
    (66.6, 55.0, 50.8, 71.0)
    """
    L = _vec(levels)
    out = {f"L{p:g}": _q7(L, 1 - p / 100.0) for p in percentiles}
    out.update({"Leq": equivalent_level(L), "Lmax": max(L), "Lmin": min(L)})
    return RichResult(payload=out)


def sound_exposure_level(levels, dt: float = 1.0) -> float:
    r"""Sound exposure level ``SEL = 10 log10(sum 10^{L_i/10} dt / T0)``, ``T0 = 1`` s, of samples ``dt`` seconds apart.

    Equivalently ``SEL = Leq + 10 log10(T / T0)`` for an event of duration ``T``.

    Examples
    --------
    >>> round(sound_exposure_level([80.0] * 10), 6)
    90.0
    """
    return _db(ssum(10 ** (v / 10) for v in _vec(levels)) * dt)


def combine_levels(levels) -> float:
    r"""Energetic sum ``10 log10(sum 10^{L_i/10})`` of incoherent sources.

    Examples
    --------
    >>> round(combine_levels([60.0, 60.0]), 6)
    63.0103
    """
    return _db(ssum(10 ** (v / 10) for v in _vec(levels)))


def _hours(a, b):
    return list(range(a, b)) if a < b else list(range(a, 24)) + list(range(0, b))


def day_evening_night(
    hourly,
    *,
    day=(7, 19),
    evening=(19, 23),
    night=(23, 7),
    evening_penalty: float = 5.0,
    night_penalty: float = 10.0,
) -> RichResult:
    r"""Day-evening-night indicators from 24 hourly Leq values (index = starting hour).

    - ``Lden`` (Directive 2002/49/EC Annex I): ``10 log10((1/24)(sum_day
      10^{L/10} + sum_evening 10^{(L+5)/10} + sum_night 10^{(L+10)/10}))``
      with the ``day``/``evening``/``night`` windows (default 07-19, 19-23,
      23-07); ``Lday``, ``Levening`` and ``Lnight`` are the energy means over
      each window.
    - ``Ldn`` (DNL, US EPA 1974): day 07-22, night 22-07 with +10 dB.
    - ``CNEL`` (California Code of Regulations, Title 21, section 5001): day
      07-19, evening 19-22 weighted by 3 (4.77 dB), night 22-07 weighted by 10.

    References
    ----------
    Directive 2002/49/EC of the European Parliament and of the Council
    relating to the assessment and management of environmental noise, Annex I.
    US EPA (1974). Information on Levels of Environmental Noise Requisite to
    Protect Public Health and Welfare with an Adequate Margin of Safety.
    EPA 550/9-74-004.

    Examples
    --------
    >>> r = day_evening_night([60.0] * 24)
    >>> round(r.Lden, 6), round(r.Ldn, 6), round(r.CNEL, 6)
    (66.395243, 66.409781, 66.651117)
    """
    L = _vec(hourly)
    if len(L) != 24:
        raise ValueError("hourly must hold 24 values")
    E = [10 ** (v / 10) for v in L]

    def emean(hs):
        return _db(ssum(E[h] for h in hs) / len(hs))

    hd, he, hn = _hours(*day), _hours(*evening), _hours(*night)
    lden = _db(
        (
            ssum(E[h] for h in hd)
            + 10 ** (evening_penalty / 10) * ssum(E[h] for h in he)
            + 10 ** (night_penalty / 10) * ssum(E[h] for h in hn)
        )
        / 24
    )
    ldn = _db((ssum(E[h] for h in _hours(7, 22)) + 10 * ssum(E[h] for h in _hours(22, 7))) / 24)
    cnel = _db(
        (
            ssum(E[h] for h in _hours(7, 19))
            + 3 * ssum(E[h] for h in _hours(19, 22))
            + 10 * ssum(E[h] for h in _hours(22, 7))
        )
        / 24
    )
    return RichResult(
        payload={"Lday": emean(hd), "Levening": emean(he), "Lnight": emean(hn), "Lden": lden, "Ldn": ldn, "CNEL": cnel}
    )


def aircraft_dnl(sel_day, sel_night=()) -> float:
    r"""Day-night level from single-event SELs: ``DNL = 10 log10(sum_d 10^{SEL/10} + 10 sum_n 10^{SEL/10}) - 10 log10(86400)``.

    ``10 log10(86400) = 49.37`` dB normalises the daily exposure to a 24-h
    average (FAA Integrated Noise Model convention).

    References
    ----------
    FAA (2015). Order 1050.1F, Environmental Impacts: Policies and Procedures, Appendix B.

    Examples
    --------
    >>> round(aircraft_dnl([90.0] * 10, [90.0]), 6)
    53.645163
    """
    e = ssum(10 ** (v / 10) for v in _vec(sel_day)) + 10 * ssum(10 ** (v / 10) for v in _vec(sel_night))
    return _db(e) - _db(86400.0)


def atmospheric_absorption(f, temperature_c: float = 20.0, humidity: float = 70.0, pressure_kpa: float = 101.325):
    r"""Pure-tone atmospheric absorption coefficient (dB/m) of ISO 9613-1:1993, equation (5).

    With ``T0 = 293.15`` K, ``T01 = 273.16`` K and ``pr = 101.325`` kPa, the
    molar concentration of water vapour is ``h = RH (psat/pr)/(pa/pr)``,
    ``psat/pr = 10^C``, ``C = -6.8346 (T01/T)^{1.261} + 4.6151``; the oxygen and
    nitrogen relaxation frequencies are ``frO = (pa/pr)(24 + 4.04e4 h (0.02 +
    h)/(0.391 + h))`` and ``frN = (pa/pr)(T/T0)^{-1/2}(9 + 280 h exp(-4.170
    ((T/T0)^{-1/3} - 1)))``, and ``alpha = 8.686 f^2 [1.84e-11 (pr/pa)
    (T/T0)^{1/2} + (T/T0)^{-5/2}(0.01275 e^{-2239.1/T}/(frO + f^2/frO) +
    0.1068 e^{-3352.0/T}/(frN + f^2/frN))]``.

    References
    ----------
    ISO 9613-1:1993. Acoustics -- Attenuation of sound during propagation
    outdoors -- Part 1: Calculation of the absorption of sound by the atmosphere.

    Evaluated at the exact octave midband frequencies ``1000 * 10^{3k/10}``
    it reproduces ISO 9613-2 Table 2 (e.g. 20 C, 70 %: 2.8, 5.0 and 22.9 dB/km
    at 500, 1000 and 4000 Hz).

    Examples
    --------
    >>> [round(1000 * a, 2) for a in atmospheric_absorption([1000 * 10**-0.3, 1000.0, 1000 * 10**0.6])]
    [2.8, 4.98, 22.91]
    """
    T = temperature_c + 273.15
    T0, T01, pr = 293.15, 273.16, 101.325
    pa = pressure_kpa / pr
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    h = humidity * 10**C / pa
    frO = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa * (T / T0) ** -0.5 * (9 + 280 * h * math.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    out = []
    for fr in _vec(f):
        f2 = fr * fr
        out.append(
            8.686
            * f2
            * (
                1.84e-11 / pa * (T / T0) ** 0.5
                + (T / T0) ** -2.5
                * (
                    0.01275 * math.exp(-2239.1 / T) / (frO + f2 / frO)
                    + 0.1068 * math.exp(-3352.0 / T) / (frN + f2 / frN)
                )
            )
        )
    return out


def point_source_level(lw: float, d, *, alpha: float = 0.0, dc: float = 0.0, extra: float = 0.0):
    r"""ISO 9613-2 receiver level of a point source: ``Lp = Lw + Dc - (20 log10(d/d0) + 11) - alpha d - extra``, ``d0 = 1`` m.

    ``alpha`` is the absorption coefficient (dB/m), ``Dc`` the directivity
    correction (3 dB for a source over a reflecting plane, as used for wind
    turbines: ``Lp = Lw - 10 log10(2 pi d^2) - alpha d``) and ``extra`` any
    further attenuation (ground, barrier, foliage).

    References
    ----------
    ISO 9613-2:1996. Acoustics -- Attenuation of sound during propagation
    outdoors -- Part 2: General method of calculation, equations (3) and (7).

    Examples
    --------
    >>> [round(v, 6) for v in point_source_level(100.0, [10.0, 100.0])]
    [69.0, 49.0]
    """
    return [lw + dc - (20 * math.log10(di) + 11) - alpha * di - extra for di in _vec(d)]


def attenuation_distance(lw: float, threshold: float, *, alpha: float = 0.0, dc: float = 0.0) -> float:
    r"""Distance at which :func:`point_source_level` falls to ``threshold`` (closed form without absorption, else Newton).

    Examples
    --------
    >>> round(attenuation_distance(100.0, 49.0), 6)
    100.0
    """
    d = 10 ** ((lw + dc - 11 - threshold) / 20)
    if alpha > 0:
        for _ in range(100):
            g = lw + dc - 20 * math.log10(d) - 11 - alpha * d - threshold
            step = g / (20 / (d * math.log(10)) + alpha)
            d = max(d + step, d / 10)
            if abs(step) < 1e-12 * d:
                break
    return d


def ground_attenuation(dp: float, hs: float, hr: float) -> RichResult:
    r"""ISO 9613-2 alternative ground attenuation (7.3.2) for A-weighted levels over mostly porous ground.

    ``A_gr = max(0, 4.8 - (2 h_m/d)(17 + 300/d))`` with mean propagation
    height ``h_m = (hs + hr)/2`` and slant distance ``d = sqrt(dp^2 + (hs -
    hr)^2)``; the companion directivity correction is ``D_Omega = 10 log10(1 +
    (dp^2 + (hs - hr)^2)/(dp^2 + (hs + hr)^2))``.

    Examples
    --------
    >>> r = ground_attenuation(200.0, 2.0, 2.0)
    >>> round(r.A_gr, 6), round(r.D_omega, 6)
    (4.43, 3.009432)
    """
    d = math.hypot(dp, hs - hr)
    hm = (hs + hr) / 2
    agr = max(0.0, 4.8 - (2 * hm / d) * (17 + 300 / d))
    dom = 10 * math.log10(1 + (dp * dp + (hs - hr) ** 2) / (dp * dp + (hs + hr) ** 2))
    return RichResult(payload={"A_gr": agr, "D_omega": dom})


def barrier_attenuation(
    d_ss: float, d_sr: float, d: float, f: float, *, a: float = 0.0, e: float = 0.0, c: float = 340.0
) -> RichResult:
    r"""ISO 9613-2 barrier screening ``Dz = 10 log10(3 + (C2/lambda) C3 z K_met)``, ``C2 = 20``.

    ``z`` is the path-length difference: ``sqrt((d_ss + d_sr)^2 + a^2) - d``
    for single diffraction and ``sqrt((d_ss + d_sr + e)^2 + a^2) - d`` for
    double diffraction over edges ``e`` apart, where ``C3 = (1 + (5
    lambda/e)^2)/(1/3 + (5 lambda/e)^2)``; ``K_met = exp(-(1/2000) sqrt(d_ss
    d_sr d/(2 z)))`` for ``z > 0``. ``Dz`` is capped at 20 dB (single) or 25
    dB (double) and is 0 when the line of sight is not interrupted (``z <=
    0``).

    Examples
    --------
    >>> round(barrier_attenuation(10.0, 10.0, 19.0, 1000.0).Dz, 6)
    17.84788
    """
    lam = c / f
    if e > 0:
        z = math.hypot(d_ss + d_sr + e, a) - d
        r5 = (5 * lam / e) ** 2
        c3, cap = (1 + r5) / (1 / 3 + r5), 25.0
    else:
        z = math.hypot(d_ss + d_sr, a) - d
        c3, cap = 1.0, 20.0
    if z <= 0:
        return RichResult(payload={"Dz": 0.0, "z": z, "K_met": 1.0})
    km = math.exp(-math.sqrt(d_ss * d_sr * d / (2 * z)) / 2000)
    return RichResult(payload={"Dz": min(cap, 10 * math.log10(3 + 20 / lam * c3 * z * km)), "z": z, "K_met": km})


_FOLIAGE = {
    63: (0.0, 0.02),
    125: (0.0, 0.03),
    250: (1.0, 0.04),
    500: (1.0, 0.05),
    1000: (1.0, 0.06),
    2000: (1.0, 0.08),
    4000: (2.0, 0.09),
    8000: (3.0, 0.12),
}


def foliage_attenuation(df: float, band: int) -> float:
    r"""Octave-band attenuation through dense foliage (ISO 9613-2 Annex A, Table A.1).

    ``df`` is the curved-path length through foliage: below 10 m no
    attenuation, 10-20 m the tabulated dB, 20-200 m the tabulated dB/m times
    ``df``, and beyond 200 m the 200 m value.

    Examples
    --------
    >>> foliage_attenuation(15.0, 4000), round(foliage_attenuation(50.0, 1000), 6), foliage_attenuation(300.0, 8000)
    (2.0, 3.0, 24.0)
    """
    if band not in _FOLIAGE:
        raise ValueError("band must be an octave midband frequency 63..8000 Hz")
    fixed, rate = _FOLIAGE[band]
    if df < 10:
        return 0.0
    if df <= 20:
        return fixed
    return rate * min(df, 200.0)


def crtn_road_noise(
    flow: float,
    *,
    speed: float = 75.0,
    heavy_pct: float = 0.0,
    gradient_pct: float = 0.0,
    distance: float = 10.0,
    receiver_height: float = 1.5,
    soft_ground: float = 0.0,
    angle: float = 180.0,
    facade: bool = False,
    period: str = "hour",
) -> RichResult:
    r"""UK Calculation of Road Traffic Noise (CRTN, 1988) prediction of ``L_A10``.

    Basic level ``42.2 + 10 log10 q`` (hourly flow) or ``29.1 + 10 log10 Q``
    (18-hour flow); speed and heavy-vehicle correction ``33 log10(V + 40 +
    500/V) + 10 log10(1 + 5p/V) - 68.8``; gradient ``0.3 G``; distance
    ``-10 log10(d'/13.5)`` with ``d' = sqrt((d + 3.5)^2 + h^2)``, ``h`` the
    receiver height above the source line (0.5 m); ground cover with
    absorbent fraction ``I`` and mean propagation height ``H = (receiver +
    0.5)/2``: ``5.2 I log10(3/(d + 3.5))`` for ``H < 0.75``, ``5.2 I
    log10((6H - 1.5)/(d + 3.5))`` for ``0.75 <= H < (d + 5)/6``, 0 above;
    angle of view ``10 log10(theta/180)``; facade ``+2.5`` dB.

    References
    ----------
    Department of Transport and Welsh Office (1988). *Calculation of Road
    Traffic Noise*. HMSO, London, Charts 3-10.

    Examples
    --------
    >>> round(crtn_road_noise(1000.0).L10, 6)
    72.198781
    """
    basic = (42.2 if period == "hour" else 29.1) + 10 * math.log10(flow)
    v, p = speed, heavy_pct
    c_speed = 33 * math.log10(v + 40 + 500 / v) + 10 * math.log10(1 + 5 * p / v) - 68.8
    h = receiver_height - 0.5
    c_dist = -10 * math.log10(math.hypot(distance + 3.5, h) / 13.5)
    H = (receiver_height + 0.5) / 2
    if soft_ground <= 0 or (distance + 5) / 6 <= H:
        c_ground = 0.0
    elif H < 0.75:
        c_ground = 5.2 * soft_ground * math.log10(3 / (distance + 3.5))
    else:
        c_ground = 5.2 * soft_ground * math.log10((6 * H - 1.5) / (distance + 3.5))
    c_angle = 10 * math.log10(angle / 180)
    parts = {
        "basic": basic,
        "speed": c_speed,
        "gradient": 0.3 * gradient_pct,
        "distance": c_dist,
        "ground": c_ground,
        "angle": c_angle,
        "facade": 2.5 if facade else 0.0,
    }
    return RichResult(payload={"L10": ssum(parts.values()), "corrections": parts})


def construction_noise(lmax50: float, distance, usage_factor: float = 100.0) -> RichResult:
    r"""FHWA Roadway Construction Noise Model: ``Lmax = Lmax@50ft - 20 log10(D/50)``, ``Leq = Lmax + 10 log10(UF/100)``.

    References
    ----------
    FHWA (2006). *Roadway Construction Noise Model User's Guide*. FHWA-HEP-05-054.

    Examples
    --------
    >>> r = construction_noise(85.0, [100.0], usage_factor=40.0)
    >>> round(r.Lmax[0], 6), round(r.Leq[0], 6)
    (78.9794, 75.0)
    """
    lmax = [lmax50 - 20 * math.log10(di / 50) for di in _vec(distance)]
    return RichResult(payload={"Lmax": lmax, "Leq": [v + 10 * math.log10(usage_factor / 100) for v in lmax]})


def vibration_propagation(ref_level: float, distance, *, ref_distance: float = 25.0, kind: str = "lv"):
    r"""FTA ground-borne vibration attenuation from a reference distance (25 ft).

    ``kind="lv"``: velocity level ``Lv(D) = Lv(ref) - 30 log10(D/Dref)``
    (VdB); ``kind="ppv"``: peak particle velocity ``PPV(D) = PPV(ref)
    (Dref/D)^1.5``.

    References
    ----------
    FTA (2018). *Transit Noise and Vibration Impact Assessment Manual*.
    FTA Report 0123, equations 7-2 and 7-3.

    Examples
    --------
    >>> vibration_propagation(94.0, [250.0]), round(vibration_propagation(0.2, [100.0], kind="ppv")[0], 6)
    ([64.0], 0.025)
    """
    D = _vec(distance)
    if kind == "lv":
        return [ref_level - 30 * math.log10(x / ref_distance) for x in D]
    if kind == "ppv":
        return [ref_level * (ref_distance / x) ** 1.5 for x in D]
    raise ValueError("kind must be lv or ppv")


def underwater_propagation(
    source_level: float, r, f_khz: float, *, transition_range: float | None = None
) -> RichResult:
    r"""Underwater transmission loss with Thorp (1967) absorption and spherical/cylindrical spreading.

    ``TL = 20 log10 r + alpha r/1000`` (spherical, ``r`` in m) or, beyond
    ``transition_range`` ``r_t``, ``20 log10 r_t + 10 log10(r/r_t) + alpha
    r/1000``; ``alpha = 0.11 f^2/(1 + f^2) + 44 f^2/(4100 + f^2) + 2.75e-4
    f^2 + 0.003`` dB/km with ``f`` in kHz. Received level ``RL = SL - TL``.

    References
    ----------
    Thorp, W. H. (1967). Analytic description of the low-frequency
    attenuation coefficient. *Journal of the Acoustical Society of America*,
    42(1), 270.

    Examples
    --------
    >>> r = underwater_propagation(200.0, [1000.0], 10.0)
    >>> round(r.alpha, 6), round(r.RL[0], 6)
    (1.18703, 138.81297)
    """
    f2 = f_khz * f_khz
    alpha = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003
    tl = []
    for x in _vec(r):
        if transition_range is None or x <= transition_range:
            s = 20 * math.log10(x)
        else:
            s = 20 * math.log10(transition_range) + 10 * math.log10(x / transition_range)
        tl.append(s + alpha * x / 1000)
    return RichResult(payload={"alpha": alpha, "TL": tl, "RL": [source_level - v for v in tl]})


_HA = {
    "aircraft": (-9.199e-5, 3.932e-2, 0.2939),
    "road": (9.868e-4, -1.436e-2, 0.5118),
    "rail": (7.239e-4, -7.851e-3, 0.1695),
}


def noise_annoyance(level, source: str = "road"):
    r"""Percentage highly annoyed from Lden (Miedema and Oudshoorn 2001; EU position paper 2002).

    ``%HA = a (L - 42)^3 + b (L - 42)^2 + c (L - 42)`` with (a, b, c) =
    aircraft (-9.199e-5, 3.932e-2, 0.2939), road (9.868e-4, -1.436e-2,
    0.5118), rail (7.239e-4, -7.851e-3, 0.1695); 0 at or below 42 dB.

    References
    ----------
    Miedema, H. M. E. and Oudshoorn, C. G. M. (2001). Annoyance from
    transportation noise: relationships with exposure metrics DNL and DENL
    and their confidence intervals. *Environmental Health Perspectives*,
    109(4), 409-416.

    Examples
    --------
    >>> [round(v, 6) for v in noise_annoyance([42.0, 60.0])]
    [0.0, 10.314778]
    """
    if source not in _HA:
        raise ValueError("source must be aircraft, road or rail")
    a, b, c = _HA[source]
    return [0.0 if v <= 42 else a * (v - 42) ** 3 + b * (v - 42) ** 2 + c * (v - 42) for v in _vec(level)]


def noise_health_risk(level, *, rr_per_10db: float = 1.08, threshold: float = 53.0, population=None) -> RichResult:
    r"""Log-linear exposure-response ``RR(L) = RR10^{(L - L0)/10}`` above ``L0`` and the population attributable fraction.

    Defaults are the WHO (2018) estimate for road traffic noise and
    incidence of ischaemic heart disease (RR 1.08 per 10 dB Lden above 53
    dB). With ``population`` counts per level, ``PAF = sum p_i (RR_i - 1) /
    (1 + sum p_i (RR_i - 1))`` with ``p_i`` the population shares.

    References
    ----------
    WHO Regional Office for Europe (2018). *Environmental Noise Guidelines
    for the European Region*.

    Examples
    --------
    >>> r = noise_health_risk([50.0, 63.0], population=[1, 1])
    >>> r.rr, round(r.paf, 6)
    ([1.0, 1.08], 0.038462)
    """
    L = _vec(level)
    rr = [rr_per_10db ** ((v - threshold) / 10) if v > threshold else 1.0 for v in L]
    out = {"rr": rr}
    if population is not None:
        w = _vec(population)
        s = ssum(w)
        x = ssum(wi / s * (ri - 1) for wi, ri in zip(w, rr))
        out["paf"] = x / (1 + x)
    return RichResult(payload=out)


def noise_map(lw, sources, receivers, *, alpha: float = 0.0, dc: float = 0.0, min_distance: float = 1.0):
    r"""Receiver levels from several point sources: energetic sum of :func:`point_source_level` terms.

    Distances below ``min_distance`` are clamped to it (the reference distance).

    Examples
    --------
    >>> [round(v, 6) for v in noise_map([100.0, 100.0], [(0, 0), (20, 0)], [(10, 0)])]
    [72.0103]
    """
    W = _vec(lw)
    S = [tuple(float(v) for v in s) for s in sources]
    out = []
    for rcv in receivers:
        q = tuple(float(v) for v in rcv)
        e = ssum(
            10 ** (point_source_level(w, [max(math.dist(s, q), min_distance)], alpha=alpha, dc=dc)[0] / 10)
            for w, s in zip(W, S)
        )
        out.append(_db(e))
    return out


def noise_zones(levels, breaks=(55.0, 60.0, 65.0, 70.0, 75.0), population=None) -> RichResult:
    r"""Classify levels into bands (Directive 2002/49/EC Annex VI reporting bands by default) and count exposure.

    Zone ``k`` holds levels in ``[breaks[k-1], breaks[k])``; zone 0 lies below
    the first break. With ``population``, the exposed population per zone is
    summed.

    Examples
    --------
    >>> r = noise_zones([50.0, 57.0, 66.0, 80.0], population=[10, 20, 30, 40])
    >>> r.zone, r.counts
    ([0, 1, 3, 5], [10.0, 20.0, 0.0, 30.0, 0.0, 40.0])
    """
    B = sorted(float(b) for b in breaks)
    z = [sum(1 for b in B if v >= b) for v in _vec(levels)]
    w = [1.0] * len(z) if population is None else _vec(population)
    counts = [0.0] * (len(B) + 1)
    for zi, wi in zip(z, w):
        counts[zi] += wi
    return RichResult(payload={"zone": z, "counts": counts, "breaks": B})


def cheatsheet() -> str:
    return "equivalent_level / day_evening_night / point_source_level / crtn_road_noise -> environmental noise."
