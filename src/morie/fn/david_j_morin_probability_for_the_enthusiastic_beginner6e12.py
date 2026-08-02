"""Sample correlation for data points, deviation form.

Implements eq (6.12) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_12"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_12(x, y):
    """Sample correlation for data points, deviation form.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.12).
    """
    value = _morin.sample_r(x, y)
    payload = {"r": value, "cov": _morin.sample_cov(x, y)}
    lines = [("r", value)]
    return RichResult(
        title="Sample correlation for data points, deviation form.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e12: Sample correlation for data points, deviation form. Morin (2016) eq (6.12)."
