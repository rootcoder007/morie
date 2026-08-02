"""First-order validity: (1+a)^n ~ e^(na) requires na^2 << 1.

Implements eq (7.23) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_23(a, n):
    """First-order validity: (1+a)^n ~ e^(na) requires na^2 << 1.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.23).
    """
    exact, approx, validity = _morin.one_plus_a_to_n(a, n, order=1)
    payload = {"exact": exact, "approx": approx, "na2": validity,
               "valid": validity < 0.1}
    lines = [("na^2", validity), ("valid", validity < 0.1)]
    return RichResult(
        title="First-order validity: (1+a)^n ~ e^(na) requires na^2 << 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e23: First-order validity: (1+a)^n ~ e^(na) requires na^2 << 1. Morin (2016) eq (7.23)."
