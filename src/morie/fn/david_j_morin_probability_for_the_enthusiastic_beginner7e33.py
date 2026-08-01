"""Difference quotient of x^n approaches n x^(n-1).

Implements eq (7.33) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33(x, n, delta):
    """Difference quotient of x^n approaches n x^(n-1).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.33).
    """
    quotient, derivative = _morin.power_derivative_quotient(x, n, delta)
    payload = {"quotient": quotient, "derivative": derivative,
               "abs_error": abs(quotient - derivative)}
    lines = [("quotient", quotient), ("n x^(n-1)", derivative)]
    return RichResult(
        title="Difference quotient of x^n approaches n x^(n-1).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e33: Difference quotient of x^n approaches n x^(n-1). Morin (2016) eq (7.33)."
