"""Sample correlation for a collection of data points.

Implements eq (6.55) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_55(x, y):
    """Sample correlation for a collection of data points.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.55).
    """
    value = _morin.sample_r(x, y)
    payload = {"r": value}
    lines = [("r", value)]
    return RichResult(
        title="Sample correlation for a collection of data points.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e55: Sample correlation for a collection of data points. Morin (2016) eq (6.55)."
