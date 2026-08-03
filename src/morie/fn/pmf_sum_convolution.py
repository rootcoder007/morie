"""pmf of the sum of two independent discrete variables (convolution).

Implements eq (3.11) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["pmf_sum_convolution"]


def pmf_sum_convolution(values_x, probs_x, values_y, probs_y):
    """pmf of the sum of two independent discrete variables (convolution).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.11).
    """
    values, probs = _morin.pmf_sum_convolution(values_x, probs_x, values_y, probs_y)
    payload = {"values": [float(v) for v in values],
               "probs": [float(p) for p in probs]}
    lines = [(f"P(S={v:g})", float(p)) for v, p in zip(values, probs)]
    return RichResult(
        title="pmf of the sum of two independent discrete variables (convolution).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e11: pmf of the sum of two independent discrete variables (convolution). Morin (2016) eq (3.11)."
