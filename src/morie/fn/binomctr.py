"""Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.3).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binomctr"]


def binomctr(x, n):
    """Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips.

    Parameters
    ----------
    x : int
        Deviation from n Heads; zero outside |x| <= n.
    n : int
        Half the number of flips, >= 0.

    Returns
    -------
    RichResult
        Keys: x, n, probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.3).
    """
    value = _morin.binomial_centered_pmf(x, n)
    payload = {"x": int(x), "n": int(n), "probability": value}
    lines = [("PB(x)", value)]
    return RichResult(
        title="Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binomctr: Centred binomial C(2n, n+x)/2^(2n). Morin (2016) eq (5.3)."
