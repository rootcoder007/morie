"""Sample correlation r = Cov(x,y)/(s_x s_y).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (6.12), (6.55).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sampr"]


def sampr(x, y):
    """Sample correlation r = Cov(x,y)/(s_x s_y).

    Uses the 1/n covariance and the 1/n standard deviations, so the
    divisors cancel and r matches the book's deviation form.

    Parameters
    ----------
    x, y : array-like
        Equal-length data vectors, n >= 2, neither constant.

    Returns
    -------
    RichResult
        Keys: r, cov.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (6.12), (6.55).
    """
    value = _morin.sample_r(x, y)
    payload = {"r": value, "cov": _morin.sample_cov(x, y)}
    lines = [("r", value)]
    return RichResult(
        title="Sample correlation r = Cov(x,y)/(s_x s_y).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sampr: Sample correlation Cov(x,y)/(s_x s_y). Morin (2016) eqs (6.12), (6.55)."
