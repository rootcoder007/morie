"""(1 + a)^n ~ e^(na).

Implements eq (7.14) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14(a, n):
    """(1 + a)^n ~ e^(na).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.14).
    """
    exact, approx, validity = _morin.one_plus_a_to_n(a, n, order=1)
    payload = {"exact": exact, "approx": approx, "na2": validity}
    lines = [("(1+a)^n", exact), ("e^(na)", approx)]
    return RichResult(
        title="(1 + a)^n ~ e^(na).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e14: (1 + a)^n ~ e^(na). Morin (2016) eq (7.14)."
