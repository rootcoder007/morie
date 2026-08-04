"""Small-x approximation e^x ~ 1 + x.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.9).
"""

import math

from ._richresult import RichResult

__all__ = ["explinapx"]


def explinapx(x):
    """Small-x approximation e^x ~ 1 + x.

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    RichResult
        Keys: exact, approx, abs_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.9).
    """
    x_f = float(x)
    exact = math.exp(x_f)
    approx = 1.0 + x_f
    payload = {"exact": exact, "approx": approx, "abs_error": abs(exact - approx)}
    lines = [("1 + x", approx), ("e^x", exact)]
    return RichResult(
        title="Small-x approximation e^x ~ 1 + x.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "explinapx: e^x ~ 1 + x for small x. Morin (2016) eq (7.9)."
