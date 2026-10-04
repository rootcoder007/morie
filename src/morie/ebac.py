"""
eBAC calculations.
Ported from internal routines for computing continuous eBAC and legal threshold binaries.
"""


def calculate_ebac(drinks: float, weight_lbs: float, hours: float, gender_constant: float) -> float:
    """
    Compute the continuous estimated Blood Alcohol Concentration (eBAC) using the standard Widmark formula.

    :param drinks: The number of standard drinks consumed (1 drink = 14 grams of alcohol).
    :type drinks: float
    :param weight_lbs: The weight of the individual in pounds.
    :type weight_lbs: float
    :param hours: Hours elapsed since drinking began.
    :type hours: float
    :param gender_constant: Widmark gender multiplier (standard is 0.73 for men, 0.66 for women).
    :type gender_constant: float
    :return: The estimated Blood Alcohol Concentration level (non-negative).
    :rtype: float
    """
    # a data-entry error must not read as a sober respondent (a negative result clamps to 0 below)
    if not weight_lbs > 0:
        raise ValueError(f"weight_lbs must be > 0, not {weight_lbs!r}")
    if drinks < 0 or hours < 0:
        raise ValueError(f"drinks and hours cannot be negative (drinks={drinks!r}, hours={hours!r})")
    if not gender_constant > 0:
        raise ValueError(f"gender_constant must be > 0, not {gender_constant!r}")

    # Widmark in US units (Matthews & Miller 1979, Addict Behav 4:55-60):
    #   BAC = (A x 5.14) / (W x r) - 0.015 h, with A in fluid ounces of ethanol and W in pounds.
    # A 14 g standard drink is 0.6 fl oz of ethanol (NIAAA); counting each drink as 1 oz
    # inflated every eBAC about 1.7x.
    ounces = drinks * 0.6
    ebac = (ounces * 5.14) / (weight_lbs * gender_constant) - (0.015 * hours)
    return max(0.0, ebac)


def is_over_legal_limit(ebac: float, limit: float = 0.08) -> int:
    """
    Determine whether an eBAC exceeds the specified legal driving limit.

    :param ebac: The calculated eBAC value to evaluate.
    :type ebac: float
    :param limit: The legal blood alcohol concentration limit, defaults to 0.08.
    :type limit: float, optional
    :return: 1 if the eBAC is greater than or equal to the limit, 0 otherwise.
    :rtype: int
    """
    return 1 if ebac >= limit else 0
