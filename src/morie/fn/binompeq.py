"""The p making P(0) = P(1) in the binomial: p = 1/(n+1).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.9).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binompeq"]


def binompeq(n):
    """The p making P(0) = P(1) in the binomial: p = 1/(n+1).

    Parameters
    ----------
    n : int
        Trials, >= 1.

    Returns
    -------
    RichResult
        Keys: p, P0, P1.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.9).
    """
    value = _morin.p_zero_equals_one(n)
    p0 = _morin.binomial_pmf(0, n, value)
    p1 = _morin.binomial_pmf(1, n, value)
    if abs(p0 - p1) > 1e-12:
        raise AssertionError("P(0) != P(1) at p = 1/(n+1)")
    payload = {"p": value, "P0": p0, "P1": p1}
    lines = [("p = 1/(n+1)", value)]
    return RichResult(
        title="The p making P(0) = P(1) in the binomial: p = 1/(n+1).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binompeq: p = 1/(n+1) equalises the binomial P(0) and P(1). Morin (2016) eq (4.9)."
