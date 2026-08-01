"""Least-squares objective S = sum (yi - (A xi + B))^2 at the optimum.

Implements eq (6.42) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_42(x, y):
    """Least-squares objective S = sum (yi - (A xi + B))^2 at the optimum.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.42).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    payload = {"A": A, "B": B, "S": S}
    lines = [("S at optimum", S)]
    return RichResult(
        title="Least-squares objective S = sum (yi - (A xi + B))^2 at the optimum.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e42: Least-squares objective S = sum (yi - (A xi + B))^2 at the optimum. Morin (2016) eq (6.42)."
