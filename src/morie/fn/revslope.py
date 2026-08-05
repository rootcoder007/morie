"""Reverse model: X predicted from Y with slope r sigma_x / sigma_y.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (6.36).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["revslope"]


def revslope(r, sigma_x, sigma_y):
    """Reverse model: X predicted from Y with slope r sigma_x / sigma_y.

    Parameters
    ----------
    r : float
        Correlation, in [-1, 1].
    sigma_x : float
        Spread of X, >= 0.
    sigma_y : float
        Spread of Y, > 0.

    Returns
    -------
    RichResult
        Keys: slope.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (6.36).
    """
    value = _morin.reverse_regression_slope(r, sigma_x, sigma_y)
    payload = {"slope": value}
    lines = [("r sigma_x / sigma_y", value)]
    return RichResult(
        title="Reverse model: X predicted from Y with slope r sigma_x / sigma_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "revslope: Reverse regression slope r sigma_x / sigma_y. Morin (2016) eq (6.36)."
