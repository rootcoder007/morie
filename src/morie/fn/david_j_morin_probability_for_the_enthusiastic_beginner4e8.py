"""Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n.

Implements eq (4.8) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_8(k, n):
    """Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.8).
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e8: Fair-coin binomial P(k Heads in n flips) = C(n,k)/2^n. Morin (2016) eq (4.8)."
