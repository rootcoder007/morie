"""Taylor series e^x = sum x^k/k!.

Implements eq (7.7) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7(x, terms=30):
    """Taylor series e^x = sum x^k/k!.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.7).
    """
    partials, closed = _morin.exp_taylor(x, terms)
    payload = {"partial_sums": partials, "e_x": closed,
               "final_error": abs(partials[-1] - closed)}
    lines = [("series", partials[-1]), ("e^x", closed)]
    return RichResult(
        title="Taylor series e^x = sum x^k/k!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e7: Taylor series e^x = sum x^k/k!. Morin (2016) eq (7.7)."
