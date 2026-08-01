"""Normal equation dS/dB = 0: residuals sum to zero at the optimum.

Implements eq (6.92) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_92(x, y):
    """Normal equation dS/dB = 0: residuals sum to zero at the optimum.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.92).
    """
    A, B, S = _morin.least_squares_fit(x, y)
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    y_a = np.atleast_1d(np.asarray(y, dtype=float))
    resid_sum = float(np.sum(y_a - (A * x_a + B)))
    payload = {"residual_sum": resid_sum, "A": A, "B": B}
    lines = [("sum of residuals", resid_sum)]
    return RichResult(
        title="Normal equation dS/dB = 0: residuals sum to zero at the optimum.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e92: Normal equation dS/dB = 0: residuals sum to zero at the optimum. Morin (2016) eq (6.92)."
