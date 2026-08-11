# morie.fn -- wave3 coordinator batch (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FAO-56 Penman-Monteith reference evapotranspiration."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["basEvap", "penman_monteith"]


def penman_monteith(T, R_n, u2, VPD, G=0.0, P=101.3):
    """
    FAO-56 Penman-Monteith reference evapotranspiration (grass), Eq. 6.

    ET0 = [0.408 D (Rn - G) + g 900/(T+273) u2 (es - ea)]
          / [D + g (1 + 0.34 u2)]

    with D the slope of the saturation vapour-pressure curve at air
    temperature T (Eq. 13), g = 0.665e-3 P the psychrometric constant
    (Eq. 8), Rn net radiation (MJ m-2 day-1), G soil heat flux
    (0 for daily steps, Eq. 42), u2 wind speed at 2 m (m/s) and
    VPD = es - ea the vapour pressure deficit (kPa).

    Printed anchor (Allen et al. 1998, Example 18, Uccle 6 July):
    T = 16.9 C, Rn = 13.28, G = 0, u2 = 2.078, VPD = 0.589,
    P = 100.1 kPa -> D = 0.122, g = 0.0666, radiative term 2.81,
    aerodynamic term 1.07, ET0 = 3.88 mm/day. Reproduced in the tests.

    Parameters
    ----------
    T : float
        Mean daily air temperature at 2 m (deg C).
    R_n : float
        Net radiation at the crop surface (MJ m-2 day-1).
    u2 : float
        Wind speed at 2 m (m/s).
    VPD : float
        Saturation vapour pressure deficit es - ea (kPa).
    G : float
        Soil heat flux density (MJ m-2 day-1); 0 for daily steps
        (FAO-56 Eq. 42).
    P : float
        Atmospheric pressure (kPa); 101.3 at sea level (Eq. 7).

    Returns
    -------
    result : RichResult
        Keys: estimate (ET0, mm/day), radiative_term,
        aerodynamic_term, delta (Eq. 13), gamma (Eq. 8), T, R_n, u2,
        VPD, G, P, method.

    References
    ----------
    Allen, R. G., Pereira, L. S., Raes, D. and Smith, M. (1998),
    "Crop evapotranspiration - Guidelines for computing crop water
    requirements", FAO Irrigation and Drainage Paper 56, FAO, Rome.
    Eq. 6 (ET0), Eq. 8 (gamma), Eq. 13 (Delta), Eq. 42 (G = 0 daily),
    Example 18 (Chapter 4). Local sources:
    fetched-wave3/fao56-x0490e0{6,7,8}.html (Chapter text incl.
    Example 18) and fetched-wave3/zotarelli-2010-fao56-step-by-step-
    AE459.pdf (equation compilation).
    """
    T = float(T)
    R_n = float(R_n)
    u2 = float(u2)
    VPD = float(VPD)
    G = float(G)
    P = float(P)
    if VPD < 0:
        raise ValueError("basEvap: VPD must be non-negative")
    if u2 < 0:
        raise ValueError("basEvap: wind speed must be non-negative")
    if P <= 0:
        raise ValueError("basEvap: pressure must be positive")
    # Eq. 13: Delta = 4098 [0.6108 exp(17.27 T / (T + 237.3))] / (T + 237.3)^2
    es_T = 0.6108 * math.exp(17.27 * T / (T + 237.3))
    delta = 4098.0 * es_T / (T + 237.3) ** 2
    # Eq. 8: gamma = 0.665e-3 P
    gamma = 0.665e-3 * P
    denom = delta + gamma * (1.0 + 0.34 * u2)
    rad = 0.408 * delta * (R_n - G) / denom
    aero = gamma * (900.0 / (T + 273.0)) * u2 * VPD / denom
    et0 = rad + aero
    return RichResult(payload={
        "estimate": et0,
        "radiative_term": rad,
        "aerodynamic_term": aero,
        "delta": delta,
        "gamma": gamma,
        "T": T, "R_n": R_n, "u2": u2, "VPD": VPD, "G": G, "P": P,
        "method": "FAO-56 Penman-Monteith ET0 (Allen et al. 1998, Eq. 6)",
    })


basEvap = penman_monteith


def cheatsheet():
    return "basEvap(T, R_n, u2, VPD, G=0, P=101.3) -> FAO-56 Eq. 6 reference ET0 (mm/day)"
