"""Small-x approximation e^x ~ 1 + x.

Implements eq (7.9) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9(x):
    """Small-x approximation e^x ~ 1 + x.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.9).
    """
    x_f = float(x)
    exact = math.exp(x_f)
    approx = 1.0 + x_f
    payload = {"exact": exact, "approx": approx,
               "abs_error": abs(exact - approx)}
    lines = [("1 + x", approx), ("e^x", exact)]
    return RichResult(
        title="Small-x approximation e^x ~ 1 + x.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e9: Small-x approximation e^x ~ 1 + x. Morin (2016) eq (7.9)."
