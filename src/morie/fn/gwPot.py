"""Global warming potentials (IPCC AR6 Table 7.SM.7)."""

from ._richresult import RichResult

__all__ = ["gwPot", "global_warming_potential"]

# Transcribed from IPCC AR6 WG1 Chapter 7 Supplementary Material,
# Table 7.SM.7 (fetched-wave3/ipcc-ar6-wg1-ch7-supplementary.pdf, text
# layer lines 3452+; cross-checked self-consistent: GWP_H = AGWP_x(H) /
# AGWP_CO2(H) reproduces every printed GWP below to its printed
# precision).  Columns: lifetime (yr), radiative efficiency
# (W m-2 ppb-1), AGWP (pW m-2 yr kg-1) and GWP at H = 20, 100, 500 yr.
_TABLE = {
    "CO2":      {"lifetime": None,  "re": 1.33e-5,
                 "agwp": {20: 0.0243, 100: 0.0895, 500: 0.314},
                 "gwp":  {20: 1.0,    100: 1.0,    500: 1.0}},
    "CH4":      {"lifetime": 11.8,  "re": 3.88e-4,
                 "agwp": {20: 1.98,  100: 2.49,  500: 2.5},
                 "gwp":  {20: 81.2,  100: 27.9,  500: 7.95}},
    "N2O":      {"lifetime": 109.0, "re": 3.2e-3,
                 "agwp": {20: 6.65,  100: 24.5,  500: 40.7},
                 "gwp":  {20: 273.0, 100: 273.0, 500: 130.0}},
    "CFC-11":   {"lifetime": 52.0,  "re": 0.291,
                 "agwp": {20: 203.0, 100: 557.0, 500: 657.0},
                 "gwp":  {20: 8320.0, 100: 6230.0, 500: 2090.0}},
    "CFC-12":   {"lifetime": 102.0, "re": 0.358,
                 "agwp": {20: 310.0, 100: 1120.0, 500: 1790.0},
                 "gwp":  {20: 12700.0, 100: 12500.0, 500: 5700.0}},
    "HFC-134a": {"lifetime": 14.0,  "re": 0.167,
                 "agwp": {20: 101.0, 100: 137.0, 500: 137.0},
                 "gwp":  {20: 4140.0, 100: 1530.0, 500: 436.0}},
    "SF6":      {"lifetime": 1000.0, "re": 0.567,
                 "agwp": {20: 442.0, 100: 2180.0, 500: 9100.0},
                 "gwp":  {20: 18200.0, 100: 24300.0, 500: 29000.0}},
}
_HORIZONS = (20, 100, 500)


def gwPot(gas, horizon=100):
    """
    Global warming potential of a greenhouse gas.

    GWP_H(x) = AGWP_x(H) / AGWP_CO2(H), the ratio of the absolute
    global warming potential (time-integrated radiative forcing of a
    1 kg pulse over horizon H) of gas x to that of CO2 (AR6 WG1
    Sec. 7.6.1, Eq. 7.SM.5.2).  This function returns the ASSESSED
    AR6 values at the assessed horizons H in {20, 100, 500} yr,
    transcribed from Table 7.SM.7; it deliberately does not
    extrapolate to other horizons because the CH4 and N2O radiative
    efficiencies include horizon-dependent indirect (chemical)
    adjustments that a pure exponential-decay AGWP cannot reproduce.

    Sources
    -------
    IPCC AR6 WG1 (2021), Chapter 7 Supplementary Material, Table
    7.SM.7 (lifetimes, radiative efficiencies, AGWPs, GWPs; local
    copy fetched-wave3/ipcc-ar6-wg1-ch7-supplementary.pdf) and
    Sec. 7.SM.5 (AGWP/GWP definitions).  Lifetimes from Hodnebrog
    et al. (2020), SF6 lifetime from AR6 Sec. 2.2.4.3, as stated in
    the table caption.

    Hodnebrog, Ø., Aamaas, B., Fuglestvedt, J. S., Marston, G., Myhre,
    G., Nielsen, C. J., Sandstad, M., Shine, K. P. & Wallington, T. J.
    (2020) "Updated Global Warming Potentials and Radiative
    Efficiencies of Halocarbons and Other Weak Atmospheric Absorbers",
    *Reviews of Geophysics* 58(3), e2019RG000691,
    doi:10.1029/2019RG000691 (local copy in fetched-wave3).

    Parameters
    ----------
    gas : str
        One of "CO2", "CH4", "N2O", "CFC-11", "CFC-12", "HFC-134a",
        "SF6" (case-insensitive).
    horizon : int
        Time horizon in years; one of 20, 100 (default), 500.

    Returns
    -------
    RichResult
        Keys: estimate (the GWP), agwp, agwp_co2, gwp_from_agwp
        (the ratio, unrounded), lifetime, radiative_efficiency,
        gas, horizon.
    """
    key = str(gas).strip().upper().replace("_", "-")
    aliases = {"CFC11": "CFC-11", "CFC12": "CFC-12",
               "HFC134A": "HFC-134a", "HFC-134A": "HFC-134a"}
    key = aliases.get(key, key)
    if key not in _TABLE:
        raise ValueError("unknown gas %r; known: %s"
                         % (gas, ", ".join(sorted(_TABLE))))
    h = int(horizon)
    if h not in _HORIZONS:
        raise ValueError("horizon must be one of 20, 100, 500 "
                         "(AR6 assessed horizons)")
    row = _TABLE[key]
    agwp = row["agwp"][h]
    agwp_co2 = _TABLE["CO2"]["agwp"][h]
    return RichResult(payload={
        "estimate": row["gwp"][h],
        "agwp": agwp,
        "agwp_co2": agwp_co2,
        "gwp_from_agwp": agwp / agwp_co2,
        "lifetime": row["lifetime"],
        "radiative_efficiency": row["re"],
        "gas": key, "horizon": h,
        "method": "IPCC AR6 Table 7.SM.7 assessed GWP",
    })


# long descriptive alias (stub-era name)
global_warming_potential = gwPot


def cheatsheet():
    return "gwPot: assessed AR6 GWP_H = AGWP_x(H)/AGWP_CO2(H), H in {20,100,500}"
