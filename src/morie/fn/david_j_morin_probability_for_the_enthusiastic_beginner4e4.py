"""Centered-interval probability P(T - dT/2 <= X <= T + dT/2).

Implements eq (4.4) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(grid, density, center, width):
    """Centered-interval probability P(T - dT/2 <= X <= T + dT/2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.4).
    """
    half = float(width) / 2.0
    value = _morin.density_interval_probability(grid, density,
                                                float(center) - half,
                                                float(center) + half)
    payload = {"probability": value, "center": float(center), "width": float(width)}
    lines = [("P(centered interval)", value)]
    return RichResult(
        title="Centered-interval probability P(T - dT/2 <= X <= T + dT/2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e4: Centered-interval probability P(T - dT/2 <= X <= T + dT/2). Morin (2016) eq (4.4)."
