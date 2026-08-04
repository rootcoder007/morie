"""Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.8).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["coinpmf"]


def coinpmf(k, n):
    """Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n.

    Parameters
    ----------
    k : int
        Heads, >= 0.
    n : int
        Flips, >= 0.

    Returns
    -------
    RichResult
        Keys: k, n, probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.8).
    """
    value = _morin.binomial_pmf(k, n, 0.5)
    payload = {"k": int(k), "n": int(n), "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "coinpmf: P(k Heads in n fair flips) = C(n,k)/2^n. Morin (2016) eq (4.8)."
