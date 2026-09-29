# morie.fn -- function file (rootcoder007/morie)
"""Soil process indices: temperature correction of electrical conductivity, caesium-137 inventory
models of soil redistribution, IPCC Tier 1 soil organic carbon stocks and sequestration rates,
and the Revised Wind Erosion Equation."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ec25_correction", "cesium_redistribution", "ipcc_soil_carbon", "rweq_wind_erosion"]


def _one(x):
    return isinstance(x, (int, float))


def _map(f, *args):
    if all(_one(a) for a in args):
        return f(*[float(a) for a in args])
    vs = [None if _one(a) else [float(v) for v in np.asarray(a, dtype=float).ravel().tolist()] for a in args]
    n = max(len(v) for v in vs if v is not None)
    cols = [v if v is not None else [float(a)] * n for v, a in zip(vs, args)]
    return [f(*r) for r in zip(*cols)]


def ec25_correction(ec, temperature, method="exponential"):
    r"""Electrical conductivity of soil water or extracts referred to 25 C.

    ``"exponential"``: ``EC25 = EC_T (0.4470 + 1.4034 exp(-T / 26.815))``
    (Sheets and Hendrickx 1995); ``"linear"``: ``EC25 = EC_T / (1 + 0.02 (T -
    25))`` (2 percent per degree).

    References
    ----------
    Sheets, K. R. and Hendrickx, J. M. H. (1995). Noninvasive soil water
    content measurement using electromagnetic induction. *Water Resources
    Research* 31, 2401-2409.

    Examples
    --------
    >>> round(ec25_correction(1.2, 15.0), 12)
    1.49895027407
    """
    if method == "exponential":
        return _map(lambda e, t: e * (0.4470 + 1.4034 * math.exp(-t / 26.815)), ec, temperature)
    if method == "linear":
        return _map(lambda e, t: e / (1 + 0.02 * (t - 25)), ec, temperature)
    raise ValueError("method must be 'exponential' or 'linear'")


def cesium_redistribution(inventory, reference, depth, bulk_density, years, p=1.0, model="proportional", h=0.0):
    r"""Soil redistribution rate (t ha^-1 yr^-1, negative = erosion) from caesium-137 inventories.

    ``X = 100 (A_ref - A) / A_ref`` is the percentage loss relative to the
    reference inventory. ``"proportional"`` (cultivated soils): ``Y = -10 d B X
    / (100 T P)`` with plough depth ``d`` (m), bulk density ``B`` (kg m^-3),
    ``T`` years since 1963 and particle-size correction ``P``.
    ``"profile"`` (uncultivated soils, exponential depth distribution with
    relaxation mass depth ``h`` kg m^-2): ``Y = -10 / (T P) h ln(1 - X/100)``.

    References
    ----------
    Walling, D. E. and He, Q. (1999). Improved models for estimating soil
    erosion rates from cesium-137 measurements. *Journal of Environmental
    Quality* 28, 611-622.

    Examples
    --------
    >>> round(cesium_redistribution(1800.0, 2400.0, 0.2, 1300.0, 50.0), 10)
    -13.0
    """

    def one(a, aref, d, b, t):
        x = 100.0 * (aref - a) / aref
        if model == "proportional":
            return -10.0 * d * b * x / (100.0 * t * p)
        if model == "profile":
            return 10.0 / (t * p) * h * math.log(1 - x / 100.0)
        raise ValueError("model must be 'proportional' or 'profile'")

    return _map(one, inventory, reference, depth, bulk_density, years)


def ipcc_soil_carbon(soc_ref, f_lu, f_mg, f_i, area=1.0, soc_initial=None, years=20.0):
    r"""IPCC Tier 1 soil organic carbon stock and annual stock change.

    ``SOC = SOC_ref F_LU F_MG F_I A`` (t C) from the reference stock (t C
    ha^-1, 0-30 cm), land-use, management and input factors and area ``A``
    (ha); with the initial stock the annual change ``(SOC - SOC_0) / D`` over
    the default transition time ``D = 20`` years (positive = sequestration),
    also given in t CO2 (``44 / 12``).

    References
    ----------
    IPCC (2006, 2019 refinement). *Guidelines for National Greenhouse Gas
    Inventories*, Vol. 4, chapter 2, eqs 2.24-2.25.

    Examples
    --------
    >>> r = ipcc_soil_carbon(60.0, 0.69, 1.1, 1.0, area=10.0, soc_initial=455.4)
    >>> round(r.stock, 10), round(r.annual_change, 10)
    (455.4, 0.0)
    """
    stock = soc_ref * f_lu * f_mg * f_i * area
    out = {"stock": stock}
    if soc_initial is not None:
        ch = (stock - soc_initial) / years
        out["annual_change"] = ch
        out["annual_co2"] = ch * 44.0 / 12.0
    return RichResult(payload=out)


def rweq_wind_erosion(
    wf, ef, scf, k_prime, cog, field_length=None, sand=None, silt=None, clay=None, om=None, caco3=None
):
    r"""Revised Wind Erosion Equation (Fryrear et al. 1998): maximum transport and field transport.

    ``Q_max = 109.8 (WF EF SCF K' COG)`` (kg m^-1) and the critical field
    length ``s = 150.71 (WF EF SCF K' COG)^-0.3711`` (m); at field length
    ``x`` the transport is ``Q(x) = Q_max (1 - exp(-(x/s)^2))`` and the soil
    loss ``2 x / s^2 Q_max exp(-(x/s)^2)`` (kg m^-2). ``ef`` or ``scf`` may be
    ``None`` to compute them from texture (percent), organic matter and
    CaCO3: ``EF = (29.09 + 0.31 Sa + 0.17 Si + 0.33 Sa/Cl - 2.59 OM - 0.95
    CaCO3) / 100`` and ``SCF = 1 / (1 + 0.0066 Cl^2 + 0.021 OM^2)``.

    References
    ----------
    Fryrear, D. W., Saleh, A., Bilbro, J. D., Schomberg, H. M., Stout, J. E. and
    Zobeck, T. M. (1998). *Revised Wind Erosion Equation (RWEQ)*. USDA-ARS
    Technical Bulletin 1.

    Examples
    --------
    >>> r = rweq_wind_erosion(20.0, 0.4, 0.6, 0.8, 0.9, field_length=200.0)
    >>> round(r.q_max, 10), round(r.critical_length, 10)
    (379.4688, 95.1212080473)
    """
    if ef is None:
        ef = (29.09 + 0.31 * sand + 0.17 * silt + 0.33 * sand / clay - 2.59 * om - 0.95 * caco3) / 100.0
    if scf is None:
        scf = 1.0 / (1.0 + 0.0066 * clay * clay + 0.021 * om * om)
    prod = wf * ef * scf * k_prime * cog
    qmax = 109.8 * prod
    s = 150.71 * prod**-0.3711
    out = {"q_max": qmax, "critical_length": s, "ef": ef, "scf": scf}
    if field_length is not None:
        x = float(field_length)
        out["transport"] = qmax * (1 - math.exp(-((x / s) ** 2)))
        out["soil_loss"] = 2 * x / (s * s) * qmax * math.exp(-((x / s) ** 2))
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "ec25_correction / cesium_redistribution / ipcc_soil_carbon / rweq_wind_erosion -> soil processes."
