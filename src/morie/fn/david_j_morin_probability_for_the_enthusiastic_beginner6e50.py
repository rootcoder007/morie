"""Reverse least squares: x = C y + D minimizing horizontal residuals.

Implements eq (6.50) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_50(x, y):
    """Reverse least squares: x = C y + D minimizing horizontal residuals.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.50).
    """
    C, D, S = _morin.least_squares_fit(y, x)
    payload = {"C": C, "D": D, "S": S}
    lines = [("C", C), ("D", D)]
    return RichResult(
        title="Reverse least squares: x = C y + D minimizing horizontal residuals.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e50: Reverse least squares: x = C y + D minimizing horizontal residuals. Morin (2016) eq (6.50)."
