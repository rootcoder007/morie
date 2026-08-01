"""Excess-score factor sqrt((1-r)/(1+r)).

Implements eq (6.81) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_81"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_81(r):
    """Excess-score factor sqrt((1-r)/(1+r)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.81).
    """
    value = _morin.excess_score_factor(r)
    payload = {"factor": value}
    lines = [("sqrt((1-r)/(1+r))", value)]
    return RichResult(
        title="Excess-score factor sqrt((1-r)/(1+r)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e81: Excess-score factor sqrt((1-r)/(1+r)). Morin (2016) eq (6.81)."
