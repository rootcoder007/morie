"""P(t, dt) = e^(-lambda t) lambda dt.

Implements eq (4.25) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_25(t, dt, lam):
    """P(t, dt) = e^(-lambda t) lambda dt.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.25).
    """
    value = _morin.exponential_interval_probability(t, dt, lam)
    payload = {"probability": value}
    lines = [("P(t, dt)", value)]
    return RichResult(
        title="P(t, dt) = e^(-lambda t) lambda dt.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e25: P(t, dt) = e^(-lambda t) lambda dt. Morin (2016) eq (4.25)."
