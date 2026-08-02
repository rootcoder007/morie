"""Poisson pmf sums to 1 via the exponential series.

Implements eq (7.11) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_11(a, kmax=200):
    """Poisson pmf sums to 1 via the exponential series.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.11).
    """
    a_f = float(a)
    if a_f < 0:
        raise ValueError("a must be >= 0")
    total = sum(_morin.poisson_pmf(k, a_f) for k in range(int(kmax)))
    payload = {"total": total, "error": abs(total - 1.0)}
    lines = [("sum P(k)", total)]
    return RichResult(
        title="Poisson pmf sums to 1 via the exponential series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e11: Poisson pmf sums to 1 via the exponential series. Morin (2016) eq (7.11)."
