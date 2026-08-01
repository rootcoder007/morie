"""Group-average relation: yavg = m xavg.

Implements eq (6.39) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39(m, xavg):
    """Group-average relation: yavg = m xavg.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.39).
    """
    value = float(m) * float(xavg)
    payload = {"yavg": value}
    lines = [("yavg", value)]
    return RichResult(
        title="Group-average relation: yavg = m xavg.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e39: Group-average relation: yavg = m xavg. Morin (2016) eq (6.39)."
