"""Expanded least-squares objective (same optimum).

Implements eq (6.43) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_43"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_43(x, y):
    """Expanded least-squares objective (same optimum).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.43).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    payload = {"A": A, "B": B, "S": S}
    lines = [("S", S)]
    return RichResult(
        title="Expanded least-squares objective (same optimum).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e43: Expanded least-squares objective (same optimum). Morin (2016) eq (6.43)."
