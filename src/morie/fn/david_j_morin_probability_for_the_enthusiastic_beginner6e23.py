"""Best prediction = mean (summary statement).

Implements eq (6.23) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_23"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_23(y):
    """Best prediction = mean (summary statement).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.23).
    """
    mu, mse = _morin.best_constant_predictor(y)
    payload = {"best_prediction": mu, "mse": mse}
    lines = [("best prediction", mu)]
    return RichResult(
        title="Best prediction = mean (summary statement).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e23: Best prediction = mean (summary statement). Morin (2016) eq (6.23)."
