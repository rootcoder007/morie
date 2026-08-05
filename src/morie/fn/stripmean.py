"""Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.74).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["stripmean"]


def stripmean(r, sigma_x, sigma_y, y0):
    """Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.

    Parameters
    ----------
    r : float
        Correlation, in [-1, 1].
    sigma_x : float
        Spread of X, >= 0.
    sigma_y : float
        Spread of Y, > 0.
    y0 : float
        The y value defining the strip.

    Returns
    -------
    RichResult
        Keys: x, slope.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.74).
    """
    slope = _morin.reverse_regression_slope(r, sigma_x, sigma_y)
    value = slope * float(y0)
    payload = {"x": value, "slope": slope}
    lines = [("x", value)]
    return RichResult(
        title="Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "stripmean: Strip mean x = (r sigma_x/sigma_y) y0. Morin (2016) eq (6.74)."
