"""Fractional improvement of prediction: 1 - r^2.

Implements eq (6.27) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_27"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_27(r):
    """Fractional improvement of prediction: 1 - r^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.27).
    """
    value = _morin.prediction_improvement(r)
    payload = {"r": float(r), "mse_fraction_remaining": value}
    lines = [("1 - r^2", value)]
    return RichResult(
        title="Fractional improvement of prediction: 1 - r^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e27: Fractional improvement of prediction: 1 - r^2. Morin (2016) eq (6.27)."
