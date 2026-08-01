"""Binomial variance E(k^2) - (np)^2 = npq.

Implements eq (4.67) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_67(n, p):
    """Binomial variance E(k^2) - (np)^2 = npq.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.67).
    """
    second = _morin.binomial_second_moment(n, p)
    mean = _morin.binomial_mean(n, p)
    value = second - mean ** 2
    direct = _morin.binomial_variance(n, p)
    if abs(value - direct) > 1e-9 * max(1.0, direct):
        raise AssertionError("moment identity disagrees with npq")
    payload = {"variance": value}
    lines = [("npq", value)]
    return RichResult(
        title="Binomial variance E(k^2) - (np)^2 = npq.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e67: Binomial variance E(k^2) - (np)^2 = npq. Morin (2016) eq (4.67)."
