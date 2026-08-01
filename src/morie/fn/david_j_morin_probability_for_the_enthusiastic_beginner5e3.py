"""Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips.

Implements eq (5.3) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3(x, n):
    """Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.3).
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e3: Centered binomial PB(x) = C(2n, n+x)/2^(2n) for 2n fair flips. Morin (2016) eq (5.3)."
