"""Var(X + Y) = Var(X) + Var(Y) for independent X, Y.

Implements eq (3.25) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_25(var_x, var_y):
    """Var(X + Y) = Var(X) + Var(Y) for independent X, Y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.25).
    """
    value = _morin.var_sum_independent([var_x, var_y])
    payload = {"var_sum": value}
    lines = [("Var(X+Y)", value)]
    return RichResult(
        title="Var(X + Y) = Var(X) + Var(Y) for independent X, Y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e25: Var(X + Y) = Var(X) + Var(Y) for independent X, Y. Morin (2016) eq (3.25)."
