"""sigma of one biased coin flip: sqrt(pq).

Implements eq (3.46) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_46"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_46(p):
    """sigma of one biased coin flip: sqrt(pq).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.46).
    """
    value = _morin.sd_bernoulli(p)
    payload = {"p": float(p), "sd": value}
    lines = [("sqrt(pq)", value)]
    return RichResult(
        title="sigma of one biased coin flip: sqrt(pq).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e46: sigma of one biased coin flip: sqrt(pq). Morin (2016) eq (3.46)."
