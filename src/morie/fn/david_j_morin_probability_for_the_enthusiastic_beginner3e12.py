"""E(X + Y) from the convolution equals E(X) + E(Y).

Implements eq (3.12) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12(values_x, probs_x, values_y, probs_y):
    """E(X + Y) from the convolution equals E(X) + E(Y).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.12).
    """
    values, probs = _morin.pmf_sum_convolution(values_x, probs_x, values_y, probs_y)
    e_sum = _morin.pmf_expectation(values, probs)
    e_parts = (_morin.pmf_expectation(values_x, probs_x)
               + _morin.pmf_expectation(values_y, probs_y))
    if abs(e_sum - e_parts) > 1e-9:
        raise AssertionError("E(X+Y) != E(X) + E(Y)")
    payload = {"e_sum": e_sum, "e_x_plus_e_y": e_parts}
    lines = [("E(X+Y)", e_sum)]
    return RichResult(
        title="E(X + Y) from the convolution equals E(X) + E(Y).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e12: E(X + Y) from the convolution equals E(X) + E(Y). Morin (2016) eq (3.12)."
