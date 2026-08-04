"""Difference quotient of x^n approaches n x^(n-1).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.33).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["diffquotn"]


def diffquotn(x, n, delta):
    """Difference quotient of x^n approaches n x^(n-1).

    Parameters
    ----------
    x : float
        Evaluation point.
    n : int
        Power, >= 0.
    delta : float
        Step, nonzero.

    Returns
    -------
    RichResult
        Keys: quotient, derivative, abs_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.33).
    """
    quotient, derivative = _morin.power_derivative_quotient(x, n, delta)
    payload = {
        "quotient": quotient,
        "derivative": derivative,
        "abs_error": abs(quotient - derivative),
    }
    lines = [("quotient", quotient), ("n x^(n-1)", derivative)]
    return RichResult(
        title="Difference quotient of x^n approaches n x^(n-1).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "diffquotn: ((x+d)^n - x^n)/d -> n x^(n-1). Morin (2016) eq (7.33)."
