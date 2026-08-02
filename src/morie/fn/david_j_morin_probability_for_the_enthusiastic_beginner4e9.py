"""p making P(0) = P(1) in the binomial: p = 1/(n+1).

Implements eq (4.9) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9(n):
    """p making P(0) = P(1) in the binomial: p = 1/(n+1).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.9).
    """
    value = _morin.p_zero_equals_one(n)
    p0 = _morin.binomial_pmf(0, n, value)
    p1 = _morin.binomial_pmf(1, n, value)
    if abs(p0 - p1) > 1e-12:
        raise AssertionError("P(0) != P(1) at p = 1/(n+1)")
    payload = {"p": value, "P0": p0, "P1": p1}
    lines = [("p = 1/(n+1)", value)]
    return RichResult(
        title="p making P(0) = P(1) in the binomial: p = 1/(n+1).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e9: p making P(0) = P(1) in the binomial: p = 1/(n+1). Morin (2016) eq (4.9)."
