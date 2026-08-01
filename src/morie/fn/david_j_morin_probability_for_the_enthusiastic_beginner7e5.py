"""Gaussian-approximation domain check: x must be << sqrt(n).

Implements eq (7.5) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_5(x, n):
    """Gaussian-approximation domain check: x must be << sqrt(n).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.5).
    """
    n_i = int(n)
    if n_i < 1:
        raise ValueError("n must be >= 1")
    ratio = abs(float(x)) / math.sqrt(n_i)
    payload = {"ratio": ratio, "well_inside": ratio < 0.1}
    lines = [("x/sqrt(n)", ratio)]
    return RichResult(
        title="Gaussian-approximation domain check: x must be << sqrt(n).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e5: Gaussian-approximation domain check: x must be << sqrt(n). Morin (2016) eq (7.5)."
