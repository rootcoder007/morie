"""Binomial pmf for n dice rolls with success probability p = 1/b.

Implements eq (4.32) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32(k, n, b):
    """Binomial pmf for n dice rolls with success probability p = 1/b.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.32).
    """
    b_f = float(b)
    if b_f < 1:
        raise ValueError("b must be >= 1")
    value = _morin.binomial_pmf(k, n, 1.0 / b_f)
    payload = {"k": int(k), "n": int(n), "p": 1.0 / b_f, "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Binomial pmf for n dice rolls with success probability p = 1/b.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e32: Binomial pmf for n dice rolls with success probability p = 1/b. Morin (2016) eq (4.32)."
