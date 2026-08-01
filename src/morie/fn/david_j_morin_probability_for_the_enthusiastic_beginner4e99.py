"""Typos worked example: P(0) = e^(-a), a = 7 gives ~0.1%.

Implements eq (4.99) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_99(a=7.0):
    """Typos worked example: P(0) = e^(-a), a = 7 gives ~0.1%.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.99).
    """
    value = _morin.poisson_pmf(0, a)
    payload = {"a": float(a), "p_zero": value}
    lines = [("P(0)", value)]
    return RichResult(
        title="Typos worked example: P(0) = e^(-a), a = 7 gives ~0.1%.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e99: Typos worked example: P(0) = e^(-a), a = 7 gives ~0.1%. Morin (2016) eq (4.99)."
