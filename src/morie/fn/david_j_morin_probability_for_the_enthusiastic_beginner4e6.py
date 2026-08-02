"""Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).

Implements eq (4.6) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_6(k, n, p):
    """Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.6).
    """
    value = _morin.binomial_pmf(k, n, p)
    payload = {"k": int(k), "n": int(n), "p": float(p), "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e6: Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k). Morin (2016) eq (4.6)."
