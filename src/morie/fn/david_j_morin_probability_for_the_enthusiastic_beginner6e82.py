"""The two intercept forms agree: <y> - A<x> equals the ratio form.

Implements eq (6.82) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82(x, y):
    """The two intercept forms agree: <y> - A<x> equals the ratio form.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.82).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    payload = {"B": B, "A": A}
    lines = [("B (both forms)", B)]
    return RichResult(
        title="The two intercept forms agree: <y> - A<x> equals the ratio form.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e82: The two intercept forms agree: <y> - A<x> equals the ratio form. Morin (2016) eq (6.82)."
