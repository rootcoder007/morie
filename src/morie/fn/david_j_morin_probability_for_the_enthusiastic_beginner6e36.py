"""Reverse model: X predicted from Y with slope r sigma_x / sigma_y.

Implements eq (6.36) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_36(r, sigma_x, sigma_y):
    """Reverse model: X predicted from Y with slope r sigma_x / sigma_y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.36).
    """
    value = _morin.reverse_regression_slope(r, sigma_x, sigma_y)
    payload = {"slope": value}
    lines = [("r sigma_x / sigma_y", value)]
    return RichResult(
        title="Reverse model: X predicted from Y with slope r sigma_x / sigma_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e36: Reverse model: X predicted from Y with slope r sigma_x / sigma_y. Morin (2016) eq (6.36)."
