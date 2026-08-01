"""Expectation of a continuous distribution: integral of x rho(x) dx.

Implements eq (4.55) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_55"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_55(grid, density):
    """Expectation of a continuous distribution: integral of x rho(x) dx.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.55).
    """
    value = _morin.density_expectation(grid, density)
    payload = {"expectation": value}
    lines = [("E(X)", value)]
    return RichResult(
        title="Expectation of a continuous distribution: integral of x rho(x) dx.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e55: Expectation of a continuous distribution: integral of x rho(x) dx. Morin (2016) eq (4.55)."
