"""Stirling approximation: P(n Heads in 2n flips) ~ 1/sqrt(pi n).

Implements eq (2.66) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66(n):
    """Stirling approximation: P(n Heads in 2n flips) ~ 1/sqrt(pi n).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.66).
    """
    approx = _morin.stirling_half_heads(n)
    exact = _morin.exact_half_heads(n)
    payload = {"n": int(n), "approx": approx, "exact": exact,
               "relative_error": abs(approx - exact) / exact}
    lines = [("1/sqrt(pi n)", approx), ("exact", exact)]
    return RichResult(
        title="Stirling approximation: P(n Heads in 2n flips) ~ 1/sqrt(pi n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e66: Stirling approximation: P(n Heads in 2n flips) ~ 1/sqrt(pi n). Morin (2016) eq (2.66)."
