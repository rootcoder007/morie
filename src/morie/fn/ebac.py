# morie.fn -- function file (rootcoder007/morie)
"""Estimated Blood Alcohol Concentration (eBAC) via Widmark formula."""

from __future__ import annotations


def calculate_ebac(
    drinks: float,
    weight_lbs: float,
    hours: float,
    gender_constant: float,
) -> float:
    """Compute continuous eBAC using the standard Widmark formula.

    Parameters
    ----------
    drinks : float
        Number of standard drinks consumed (1 drink = 14 g alcohol).
    weight_lbs : float
        Body weight in pounds.
    hours : float
        Hours elapsed since drinking began.
    gender_constant : float
        Widmark gender multiplier (0.73 male, 0.66 female).

    Returns
    -------
    float
        Estimated BAC (non-negative).
    """
    if weight_lbs <= 0:
        return 0.0
    # Widmark in US units (Matthews & Miller 1979, Addict Behav 4:55-60):
    #   BAC = (A x 5.14) / (W x r) - 0.015 h, with A in fluid ounces of ethanol and W in pounds.
    # A 14 g standard drink is 0.6 fl oz of ethanol (NIAAA); counting each drink as 1 oz
    # inflated every eBAC about 1.7x.
    ounces = drinks * 0.6
    ebac = (ounces * 5.14) / (weight_lbs * gender_constant) - (0.015 * hours)
    return max(0.0, ebac)


ebac = calculate_ebac


def cheatsheet() -> str:
    return "calculate_ebac({}) -> Estimated Blood Alcohol Concentration (eBAC) via Widmark for"


# compact alias per ledger/NAMING.md
calculateebac = calculate_ebac
