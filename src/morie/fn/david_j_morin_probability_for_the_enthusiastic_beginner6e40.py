"""Regression to the mean: yavg = r^2 y1.

Implements eq (6.40) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40(r, y1):
    """Regression to the mean: yavg = r^2 y1.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.40).
    """
    value = _morin.regression_to_mean_factor(r) * float(y1)
    payload = {"yavg": value, "factor": _morin.regression_to_mean_factor(r)}
    lines = [("r^2 y1", value)]
    return RichResult(
        title="Regression to the mean: yavg = r^2 y1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e40: Regression to the mean: yavg = r^2 y1. Morin (2016) eq (6.40)."
