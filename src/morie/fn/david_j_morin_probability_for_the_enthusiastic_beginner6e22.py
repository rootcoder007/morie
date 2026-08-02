"""The constant predictor minimizing squared error is the mean.

Implements eq (6.22) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22(y):
    """The constant predictor minimizing squared error is the mean.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.22).
    """
    mu, mse = _morin.best_constant_predictor(y)
    payload = {"best_prediction": mu, "mse": mse}
    lines = [("y_p = mean", mu)]
    return RichResult(
        title="The constant predictor minimizing squared error is the mean.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e22: The constant predictor minimizing squared error is the mean. Morin (2016) eq (6.22)."
