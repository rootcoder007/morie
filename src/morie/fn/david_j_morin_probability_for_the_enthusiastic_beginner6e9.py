"""r = Cov(X,Y)/(sigma_x sigma_y) for data.

Implements eq (6.9) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_9(x, y):
    """r = Cov(X,Y)/(sigma_x sigma_y) for data.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.9).
    """
    value = _morin.sample_r(x, y)
    payload = {"r": value}
    lines = [("r", value)]
    return RichResult(
        title="r = Cov(X,Y)/(sigma_x sigma_y) for data.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e9: r = Cov(X,Y)/(sigma_x sigma_y) for data. Morin (2016) eq (6.9)."
