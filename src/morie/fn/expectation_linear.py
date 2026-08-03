"""Linearity of expectation: E(aX + bY + c) = aE(X) + bE(Y) + c.

Implements eq (3.13) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["expectation_linear"]


def expectation_linear(a, e_x, b, e_y, c):
    """Linearity of expectation: E(aX + bY + c) = aE(X) + bE(Y) + c.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.13).
    """
    value = _morin.expectation_linear(a, e_x, b, e_y, c)
    payload = {"expectation": value}
    lines = [("E(aX + bY + c)", value)]
    return RichResult(
        title="Linearity of expectation: E(aX + bY + c) = aE(X) + bE(Y) + c.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e13: Linearity of expectation: E(aX + bY + c) = aE(X) + bE(Y) + c. Morin (2016) eq (3.13)."
