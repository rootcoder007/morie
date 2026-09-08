# morie.fn -- wave3 coordinator batch (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CO2 radiative forcing (IPCC AR6 / Meinshausen 2020; Myhre 1998 option)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["co2RF", "radiative_forcing_co2"]

# Table 7.SM.1 coefficients (IPCC AR6 WG1 Chapter 7 Supplementary
# Material, p. 3; Meinshausen et al. 2020 fit to the Oslo LBL cases)
_A1 = -2.4785e-7   # W m-2 ppm-2
_B1 = 7.5906e-4    # W m-2 ppm-1
_C1 = -2.1492e-3   # W m-2 ppb-1/2 (N2O band-overlap term)
_D1 = 5.2488       # W m-2
_C0_FIT = 277.15   # ppm (table reference concentration)


def radiative_forcing_co2(C, C0=_C0_FIT, N=273.87, method="ar6",
                          erf_adjustment=False):
    """
    Stratospheric-temperature-adjusted radiative forcing (SARF) of CO2.

    method="ar6" (default): the Meinshausen et al. (2020) simplified
    expression as adopted by IPCC AR6 (Table 7.SM.1):

        C_amax  = C0 - b1 / (2 a1)
        alpha'  = d1 - b1^2/(4 a1)              for C > C_amax
                  d1 + a1 (C-C0)^2 + b1 (C-C0)  for C0 < C < C_amax
                  d1                            for C < C0
        SARF    = (alpha' + c1 sqrt(N)) ln(C / C0)

    with N the N2O concentration (ppb) entering the band-overlap term.
    Printed anchor (7.SM.1.2): a doubling from the 1750 baseline
    (C0 = 278.3 ppm, N = 270.1 ppb) gives SARF 3.75 W m-2; ERF adds
    +5% tropospheric adjustment to 3.93 W m-2. Reproduced in tests.

    method="myhre1998": the older logarithmic expression
    SARF = 5.35 ln(C/C0) (Myhre et al. 1998; used through AR5) -- the
    stub's documented formula, kept as an option and labelled.

    Parameters
    ----------
    C : float
        CO2 concentration (ppm).
    C0 : float
        Reference concentration (ppm). Table value 277.15; use 278.3
        for forcing relative to 1750 per AR6 Ch. 7.
    N : float
        N2O concentration (ppb) for the overlap term (AR6 method only).
        Table reference 273.87; 1750 value 270.1.
    method : {"ar6", "myhre1998"}
    erf_adjustment : bool
        If True, add the AR6 +5% tropospheric adjustment (ERF).

    Returns
    -------
    result : RichResult
        Keys: estimate (W m-2), sarf, alpha_prime (ar6), method_used,
        C, C0, N, erf_adjustment.

    References
    ----------
    IPCC AR6 WG1 (2021), Chapter 7 Supplementary Material,
    Table 7.SM.1 and Section 7.SM.1.2 (local:
    fetched-wave3/ipcc-ar6-wg1-ch7-supplementary.pdf, p. 3, read from
    the rendered page); Meinshausen, M. et al. (2020), Geoscientific
    Model Development 13, 3571-3605 (the fit); Myhre, G., Highwood,
    E. J., Shine, K. P. and Stordal, F. (1998), Geophysical Research
    Letters 25(14), 2715-2718 (the 5.35 ln(C/C0) expression).
    """
    C = float(C)
    C0 = float(C0)
    N = float(N)
    if C <= 0 or C0 <= 0:
        raise ValueError("co2RF: concentrations must be positive")
    if method == "myhre1998":
        sarf = 5.35 * math.log(C / C0)
        alpha = None
    elif method == "ar6":
        if N < 0:
            raise ValueError("co2RF: N2O concentration must be non-negative")
        c_amax = C0 - _B1 / (2.0 * _A1)
        if C > c_amax:
            alpha = _D1 - _B1 * _B1 / (4.0 * _A1)
        elif C > C0:
            alpha = _D1 + _A1 * (C - C0) ** 2 + _B1 * (C - C0)
        else:
            alpha = _D1
        sarf = (alpha + _C1 * math.sqrt(N)) * math.log(C / C0)
    else:
        raise ValueError("co2RF: method must be 'ar6' or 'myhre1998'")
    est = sarf * 1.05 if erf_adjustment else sarf
    return RichResult(payload={
        "estimate": est,
        "sarf": sarf,
        "alpha_prime": alpha,
        "method_used": method,
        "C": C, "C0": C0, "N": N,
        "erf_adjustment": bool(erf_adjustment),
        "method": ("CO2 SARF, Meinshausen 2020 / AR6 Table 7.SM.1"
                   if method == "ar6" else
                   "CO2 SARF, Myhre 1998 5.35 ln(C/C0)"),
    })


co2RF = radiative_forcing_co2


def cheatsheet():
    return "co2RF(C, C0, N) -> AR6/Meinshausen-2020 CO2 SARF (Table 7.SM.1); method=myhre1998 for 5.35 ln(C/C0)"
