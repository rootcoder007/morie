"""Least-squares slope A = (<xy> - <x><y>)/(<x^2> - <x>^2).

Implements eq (6.47) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_47(x, y):
    """Least-squares slope A = (<xy> - <x><y>)/(<x^2> - <x>^2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.47).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    payload = {"A": A, "B": B}
    lines = [("A", A)]
    return RichResult(
        title="Least-squares slope A = (<xy> - <x><y>)/(<x^2> - <x>^2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e47: Least-squares slope A = (<xy> - <x><y>)/(<x^2> - <x>^2). Morin (2016) eq (6.47)."
