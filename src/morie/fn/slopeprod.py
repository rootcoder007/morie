"""The two regression slopes multiply to r^2.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.53).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["slopeprod"]


def slopeprod(x, y):
    """The two regression slopes multiply to r^2.

    A is the slope of y on x, C the slope of x on y; A C = r^2 is
    asserted, so a sign or divisor error fails loudly.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2.

    Returns
    -------
    RichResult
        Keys: r, slope_product_AC.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.53).
    """
    value = _morin.sample_r(x, y)
    A, C, r = _morin.regression_slope_product(x, y)
    payload = {"r": value, "slope_product_AC": A * C}
    lines = [("r", value), ("A*C = r^2", A * C)]
    return RichResult(
        title="The two regression slopes multiply to r^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "slopeprod: The forward and reverse regression slopes multiply to r^2. Morin (2016) eq (6.53)."
