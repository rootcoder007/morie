"""Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n).

Implements eq (4.71) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["hypergeometric_pmf"]


def hypergeometric_pmf(k, N, K, n):
    """Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.71).

    Examples
    --------
    >>> round(hypergeometric_pmf(2, 20, 7, 5)["probability"], 12)
    0.387383900929
    """
    value = _morin.hypergeometric_pmf(k, N, K, n)
    payload = {"probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e71: Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n). Morin (2016) eq (4.71)."
