"""Crossing time of two exponential survivals (book: t = 9.24).

Implements eq (4.30) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["exponential_crossing_time"]


def exponential_crossing_time(rate_fast=0.2, rate_slow=0.05, ratio=4.0):
    """Crossing time of two exponential survivals (book: t = 9.24).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.30).
    """
    value = _morin.exponential_crossing_time(rate_fast, rate_slow, ratio)
    payload = {"t": value}
    lines = [("crossing time", value)]
    return RichResult(
        title="Crossing time of two exponential survivals (book: t = 9.24).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e30: Crossing time of two exponential survivals (book: t = 9.24). Morin (2016) eq (4.30)."
