"""sigma of X + Y for independent variables: sqrt(sigma_x^2 + sigma_y^2).

Implements eq (3.42) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42(sigma_x, sigma_y):
    """sigma of X + Y for independent variables: sqrt(sigma_x^2 + sigma_y^2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.42).
    """
    value = _morin.sd_sum_independent([sigma_x, sigma_y])
    payload = {"sd_sum": value}
    lines = [("sigma_{X+Y}", value)]
    return RichResult(
        title="sigma of X + Y for independent variables: sqrt(sigma_x^2 + sigma_y^2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e42: sigma of X + Y for independent variables: sqrt(sigma_x^2 + sigma_y^2). Morin (2016) eq (3.42)."
