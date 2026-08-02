"""Probability from a continuous density: P = rho(T) dT over an interval.

Implements eq (4.2) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_2(grid, density, a, b):
    """Probability from a continuous density: P = rho(T) dT over an interval.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.2).
    """
    value = _morin.density_interval_probability(grid, density, a, b)
    payload = {"probability": value, "a": float(a), "b": float(b)}
    lines = [("P(a <= X <= b)", value)]
    return RichResult(
        title="Probability from a continuous density: P = rho(T) dT over an interval.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e2: Probability from a continuous density: P = rho(T) dT over an interval. Morin (2016) eq (4.2)."
