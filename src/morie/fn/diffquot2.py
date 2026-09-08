"""Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.31).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["diffquot2"]


def diffquot2(x, delta):
    """Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.

    Parameters
    ----------
    x : float
        Evaluation point.
    delta : float
        Step, nonzero.

    Returns
    -------
    RichResult
        Keys: quotient, derivative_limit.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.31).
    """
    quotient, derivative = _morin.power_derivative_quotient(x, 2, delta)
    explicit = 2.0 * float(x) + float(delta)
    if abs(quotient - explicit) > 1e-9 * max(1.0, abs(explicit)):
        raise AssertionError("quotient != 2x + delta")
    payload = {"quotient": quotient, "derivative_limit": derivative}
    lines = [("(f(x+d)-f(x))/d", quotient), ("2x", derivative)]
    return RichResult(
        title="Difference quotient of x^2: ((x+d)^2 - x^2)/d = 2x + d.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "diffquot2: ((x+d)^2 - x^2)/d = 2x + d. Morin (2016) eq (7.31)."
