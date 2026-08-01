"""Second-order: (1+a)^n ~ e^(na) e^(-na^2/2), requires na^3 << 1.

Implements eq (7.24) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24(a, n):
    """Second-order: (1+a)^n ~ e^(na) e^(-na^2/2), requires na^3 << 1.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.24).
    """
    exact, approx, validity = _morin.one_plus_a_to_n(a, n, order=2)
    payload = {"exact": exact, "approx": approx, "na3": validity,
               "valid": validity < 0.1}
    lines = [("approx", approx), ("na^3", validity)]
    return RichResult(
        title="Second-order: (1+a)^n ~ e^(na) e^(-na^2/2), requires na^3 << 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e24: Second-order: (1+a)^n ~ e^(na) e^(-na^2/2), requires na^3 << 1. Morin (2016) eq (7.24)."
