"""Least-squares intercept B (module name from the literal 19.2 = <x^2> in eq (6.89)).

Implements eq (6.49; the 19.2 in eq (6.89)) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_19_equation_2(x=None, y=None):
    """Least-squares intercept B (module name from the literal 19.2 = <x^2> in eq (6.89)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.49; the 19.2 in eq (6.89)).
    """
    x_d = [2.0, 3.0, 3.0, 5.0, 7.0] if x is None else x
    y_d = [1.0, 1.0, 3.0, 4.0, 6.0] if y is None else y
    A, B, S = _morin.least_squares_fit(x_d, y_d)
    payload = {"A": A, "B": B}
    lines = [("B", B)]
    return RichResult(
        title="Least-squares intercept B (module name from the literal 19.2 = <x^2> in eq (6.89)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner19e2: Least-squares intercept B (module name from the literal 19.2 = <x^2> in eq (6.89)). Morin (2016) eq (6.49; the 19.2 in eq (6.89))."
