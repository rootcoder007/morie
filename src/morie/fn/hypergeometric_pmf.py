"""Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n).

Implements eq (4.71) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["hypergeometric_pmf"]


def hypergeometric_pmf(k, N, K, n):
    """Hypergeometric distribution P(k) = C(K,k)C(N-K,n-k)/C(N,n).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.71).
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
