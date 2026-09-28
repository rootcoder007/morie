# morie.fn -- function file (rootcoder007/morie)
"""Water quality: CCME and weighted-arithmetic indices, dissolved-oxygen saturation, Carlson trophic state,
irrigation suitability, dissolved and suspended solids, loads and treatment removal."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "ccme_wqi",
    "weighted_arithmetic_wqi",
    "do_saturation",
    "carlson_tsi",
    "irrigation_water_quality",
    "total_dissolved_solids",
    "suspended_solids",
    "constituent_load",
    "removal_efficiency",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def ccme_wqi(values, objectives, direction=None) -> RichResult:
    r"""CCME Water Quality Index 1.0 for a samples x variables matrix against use-specific objectives.

    ``F1`` is the percentage of variables failing at least once, ``F2`` the
    percentage of individual tests failing; each failed test's excursion is
    ``value/objective - 1`` (objective must not be exceeded, ``"max"``) or
    ``objective/value - 1`` (must not fall below, ``"min"``); ``nse = sum
    excursions / all tests``, ``F3 = nse/(0.01 nse + 0.01)`` and ``WQI = 100
    - sqrt(F1^2 + F2^2 + F3^2)/1.732``. Missing values (``nan``) are not tests.
    Categories: Excellent 95-100, Good 80-94, Fair 65-79, Marginal 45-64,
    Poor 0-44. The objectives define the use (drinking, aquatic life,
    recreation, irrigation, livestock, industrial).

    References
    ----------
    Canadian Council of Ministers of the Environment (2001). *Canadian Water
    Quality Guidelines for the Protection of Aquatic Life: CCME Water Quality
    Index 1.0, Technical Report*.

    Examples
    --------
    >>> r = ccme_wqi([[1.0, 8.0], [3.0, 5.0], [1.5, 9.0]], [2.0, 6.0], ["max", "min"])
    >>> round(r.F1, 6), round(r.F2, 6), round(r.wqi, 6), r.category
    (100.0, 33.333333, 38.841939, 'Poor')
    """
    X = [_vec(r) for r in values]
    obj = _vec(objectives)
    p = len(obj)
    dirs = ["max"] * p if direction is None else list(direction)
    failed_vars, n_tests, n_fail, exc = set(), 0, 0, 0.0
    for row in X:
        for j in range(p):
            v = row[j]
            if v != v:
                continue
            n_tests += 1
            bad = v > obj[j] if dirs[j] == "max" else v < obj[j]
            if bad:
                n_fail += 1
                failed_vars.add(j)
                exc += v / obj[j] - 1 if dirs[j] == "max" else obj[j] / v - 1
    F1 = 100.0 * len(failed_vars) / p
    F2 = 100.0 * n_fail / n_tests
    nse = exc / n_tests
    F3 = nse / (0.01 * nse + 0.01)
    wqi = 100 - math.sqrt(F1 * F1 + F2 * F2 + F3 * F3) / 1.732
    cat = (
        "Excellent"
        if wqi >= 95
        else "Good"
        if wqi >= 80
        else "Fair"
        if wqi >= 65
        else "Marginal"
        if wqi >= 45
        else "Poor"
    )
    return RichResult(payload={"F1": F1, "F2": F2, "F3": F3, "nse": nse, "wqi": wqi, "category": cat})


def weighted_arithmetic_wqi(values, standards, ideal=None) -> RichResult:
    r"""Weighted arithmetic water quality index (Brown et al. 1972; Tiwari and Mishra 1985).

    Quality ratings ``q_i = 100 (V_i - V_i0)/(S_i - V_i0)`` (ideal value
    ``V_i0``, 0 by default; 7 for pH), unit weights ``w_i = K/S_i`` with ``K =
    1/sum(1/S_i)``, and ``WQI = sum q_i w_i / sum w_i``. Values above 100
    indicate water unsuitable for the use behind the standards ``S_i``.

    References
    ----------
    Brown, R. M., McClelland, N. I., Deininger, R. A. and O'Connor, M. F.
    (1972). A water quality index -- crashing the psychological barrier.
    *Indicators of Environmental Quality*, 173-182.

    Examples
    --------
    >>> r = weighted_arithmetic_wqi([7.5, 250.0, 2.0], [8.5, 500.0, 5.0], ideal=[7.0, 0.0, 0.0])
    >>> [round(v, 6) for v in r.q], round(r.wqi, 6)
    ([33.333333, 50.0, 40.0], 37.608882)
    """
    V, S = _vec(values), _vec(standards)
    V0 = [0.0] * len(V) if ideal is None else _vec(ideal)
    q = [100 * (v - v0) / (s - v0) for v, s, v0 in zip(V, S, V0)]
    K = 1 / ssum(1 / s for s in S)
    w = [K / s for s in S]
    return RichResult(payload={"q": q, "w": w, "wqi": ssum(a * b for a, b in zip(q, w)) / ssum(w)})


def do_saturation(temperature_c, *, salinity: float = 0.0, pressure_atm: float = 1.0, measured=None) -> RichResult:
    r"""Dissolved-oxygen saturation (mg/L) of Benson and Krause (1984), as tabulated in APHA 4500-O.

    ``ln C* = -139.34411 + 1.575701e5/T - 6.642308e7/T^2 + 1.243800e10/T^3 -
    8.621949e11/T^4 - S (0.017674 - 10.754/T + 2140.7/T^2)`` (``T`` in K,
    salinity ``S`` in g/kg) at 1 atm; at pressure ``P`` (atm) ``C_p = C* P
    (1 - P_wv/P)(1 - theta P)/((1 - P_wv)(1 - theta))`` with ``ln P_wv =
    11.8571 - 3840.70/T - 216961/T^2`` and ``theta = 0.000975 - 1.426e-5 t +
    6.436e-8 t^2``. With ``measured`` DO the percent saturation is returned.

    References
    ----------
    Benson, B. B. and Krause, D. (1984). The concentration and isotopic
    fractionation of oxygen dissolved in freshwater and seawater in
    equilibrium with the atmosphere. *Limnology and Oceanography*, 29(3), 620-632.

    Examples
    --------
    >>> [round(v, 3) for v in do_saturation([0.0, 20.0, 30.0]).cs]
    [14.621, 9.092, 7.559]
    """
    out = []
    t_list = _vec(temperature_c)
    for t in t_list:
        T = t + 273.15
        lnc = (
            -139.34411
            + 1.575701e5 / T
            - 6.642308e7 / T**2
            + 1.243800e10 / T**3
            - 8.621949e11 / T**4
            - salinity * (0.017674 - 10.754 / T + 2140.7 / T**2)
        )
        c = math.exp(lnc)
        if pressure_atm != 1.0:
            pwv = math.exp(11.8571 - 3840.70 / T - 216961 / T**2)
            th = 0.000975 - 1.426e-5 * t + 6.436e-8 * t * t
            P = pressure_atm
            c = c * P * (1 - pwv / P) * (1 - th * P) / ((1 - pwv) * (1 - th))
        out.append(c)
    res = {"cs": out}
    if measured is not None:
        res["percent_saturation"] = [100 * m / c for m, c in zip(_vec(measured), out)]
    return RichResult(payload=res)


def carlson_tsi(*, secchi_m=None, chla_ugl=None, tp_ugl=None) -> RichResult:
    r"""Carlson (1977) trophic state indices from Secchi depth (m), chlorophyll-a and total phosphorus (ug/L).

    ``TSI(SD) = 60 - 14.41 ln SD``, ``TSI(Chl) = 9.81 ln Chl + 30.6``,
    ``TSI(TP) = 14.42 ln TP + 4.15``; the mean of the available indices is
    classed oligotrophic (< 40), mesotrophic (40-50), eutrophic (50-70) or
    hypereutrophic (> 70).

    References
    ----------
    Carlson, R. E. (1977). A trophic state index for lakes. *Limnology and
    Oceanography*, 22(2), 361-369.

    Examples
    --------
    >>> r = carlson_tsi(secchi_m=2.0, chla_ugl=10.0, tp_ugl=30.0)
    >>> round(r.tsi_sd, 6), round(r.tsi_chl, 6), round(r.tsi_tp, 6), r.state
    (50.011749, 53.18836, 53.195266, 'eutrophic')
    """
    out = {}
    if secchi_m is not None:
        out["tsi_sd"] = 60 - 14.41 * math.log(secchi_m)
    if chla_ugl is not None:
        out["tsi_chl"] = 9.81 * math.log(chla_ugl) + 30.6
    if tp_ugl is not None:
        out["tsi_tp"] = 14.42 * math.log(tp_ugl) + 4.15
    if not out:
        raise ValueError("give at least one of secchi_m, chla_ugl, tp_ugl")
    m = ssum(out.values()) / len(out)
    out["tsi_mean"] = m
    out["state"] = (
        "oligotrophic" if m < 40 else "mesotrophic" if m < 50 else "eutrophic" if m <= 70 else "hypereutrophic"
    )
    return RichResult(payload=out)


def irrigation_water_quality(
    *,
    na: float,
    ca: float,
    mg: float,
    k: float = 0.0,
    hco3: float = 0.0,
    co3: float = 0.0,
    ec_us_cm: float | None = None,
) -> RichResult:
    r"""Irrigation suitability indices from ionic concentrations in meq/L.

    Sodium adsorption ratio ``SAR = Na/sqrt((Ca + Mg)/2)``; residual sodium
    carbonate ``RSC = (CO3 + HCO3) - (Ca + Mg)`` (Eaton 1950); percent
    sodium ``100 (Na + K)/(Ca + Mg + Na + K)`` (Wilcox 1955); Kelly's ratio
    ``Na/(Ca + Mg)``; magnesium hazard ``100 Mg/(Ca + Mg)``; and, with
    ``ec_us_cm``, the USSL salinity class C1 (< 250), C2 (250-750), C3
    (750-2250) or C4 (> 2250 uS/cm) (Richards 1954).

    References
    ----------
    Richards, L. A. (ed.) (1954). *Diagnosis and Improvement of Saline and
    Alkali Soils*. USDA Agriculture Handbook 60.

    Examples
    --------
    >>> r = irrigation_water_quality(na=6.0, ca=3.0, mg=5.0, k=0.5, hco3=4.0, ec_us_cm=900.0)
    >>> r.sar, r.rsc, round(r.percent_na, 6), r.salinity_class
    (3.0, -4.0, 44.827586, 'C3')
    """
    out = {
        "sar": na / math.sqrt((ca + mg) / 2),
        "rsc": (co3 + hco3) - (ca + mg),
        "percent_na": 100 * (na + k) / (ca + mg + na + k),
        "kelly_ratio": na / (ca + mg),
        "magnesium_hazard": 100 * mg / (ca + mg),
    }
    if ec_us_cm is not None:
        out["salinity_class"] = (
            "C1" if ec_us_cm < 250 else "C2" if ec_us_cm <= 750 else "C3" if ec_us_cm <= 2250 else "C4"
        )
    return RichResult(payload=out)


def total_dissolved_solids(*, ec_us_cm=None, k: float = 0.64, ions=None, bicarbonate: float = 0.0) -> RichResult:
    r"""Total dissolved solids (mg/L) from conductance (``TDS = k EC``) and/or as the sum of major ions.

    The ionic sum adds the dissolved constituents (mg/L) with bicarbonate
    converted to its carbonate equivalent (``0.4917 HCO3``), the residue a
    sample leaves on evaporation (Hem 1985). ``k`` is typically 0.55-0.75.

    References
    ----------
    Hem, J. D. (1985). *Study and Interpretation of the Chemical
    Characteristics of Natural Water*. USGS Water-Supply Paper 2254, 3rd ed.

    Examples
    --------
    >>> r = total_dissolved_solids(ec_us_cm=500.0, ions=[40.0, 10.0, 20.0, 30.0, 50.0], bicarbonate=100.0)
    >>> r.tds_ec, round(r.tds_ions, 2)
    (320.0, 199.17)
    """
    out = {}
    if ec_us_cm is not None:
        out["tds_ec"] = k * ec_us_cm
    if ions is not None:
        out["tds_ions"] = ssum(_vec(ions)) + 0.4917 * bicarbonate
    return RichResult(payload=out)


def suspended_solids(
    residue_mg: float, tare_mg: float, volume_ml: float, *, ignited_mg: float | None = None
) -> RichResult:
    r"""Gravimetric total suspended solids (APHA 2540 D): ``TSS = (A - B) 1000 / V`` mg/L.

    ``A`` is filter plus dried residue (mg), ``B`` the filter tare (mg) and
    ``V`` the sample volume (mL); with the post-ignition weight ``ignited_mg``
    (APHA 2540 E) the volatile fraction ``(A - C) 1000 / V`` is returned too.

    References
    ----------
    APHA, AWWA and WEF (2017). *Standard Methods for the Examination of
    Water and Wastewater*, 23rd ed., method 2540.

    Examples
    --------
    >>> r = suspended_solids(1523.4, 1510.2, 250.0, ignited_mg=1514.6)
    >>> round(r.tss, 6), round(r.vss, 6)
    (52.8, 35.2)
    """
    out = {"tss": (residue_mg - tare_mg) * 1000 / volume_ml}
    if ignited_mg is not None:
        out["vss"] = (residue_mg - ignited_mg) * 1000 / volume_ml
        out["fss"] = out["tss"] - out["vss"]
    return RichResult(payload=out)


def constituent_load(concentration, flow, dt: float = 86400.0) -> RichResult:
    r"""Constituent load ``sum C_i Q_i dt`` and flow-weighted mean concentration ``sum C_i Q_i / sum Q_i``.

    With concentrations in mg/L, flows in m^3/s and ``dt`` in s the load is
    in g (mg/L = g/m^3); used for total nitrogen, total phosphorus and
    suspended sediment loads.

    Examples
    --------
    >>> r = constituent_load([2.0, 4.0], [1.0, 3.0], dt=1.0)
    >>> r.load, r.fwmc
    (14.0, 3.5)
    """
    C, Q = _vec(concentration), _vec(flow)
    cq = ssum(c * q for c, q in zip(C, Q))
    return RichResult(payload={"load": cq * dt, "fwmc": cq / ssum(Q)})


def removal_efficiency(influent, effluent) -> RichResult:
    r"""Treatment performance: percent removal ``100 (1 - Ce/Ci)`` and log removal value ``log10(Ci/Ce)``.

    Examples
    --------
    >>> r = removal_efficiency([200.0, 1e6], [20.0, 1e2])
    >>> r.percent, r.lrv
    ([90.0, 99.99], [1.0, 4.0])
    """
    Ci, Ce = _vec(influent), _vec(effluent)
    return RichResult(
        payload={
            "percent": [100 * (1 - e / i) for i, e in zip(Ci, Ce)],
            "lrv": [math.log10(i / e) for i, e in zip(Ci, Ce)],
        }
    )


def cheatsheet() -> str:
    return "ccme_wqi / do_saturation / carlson_tsi / irrigation_water_quality -> water quality."
