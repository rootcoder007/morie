"""Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.

Implements eq (6.74) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74(r, sigma_x, sigma_y, y0):
    """Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.74).
    """
    slope = _morin.reverse_regression_slope(r, sigma_x, sigma_y)
    value = slope * float(y0)
    payload = {"x": value, "slope": slope}
    lines = [("x", value)]
    return RichResult(
        title="Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e74: Strip mean via the upper regression line: x = (r sigma_x/sigma_y) y0. Morin (2016) eq (6.74)."
