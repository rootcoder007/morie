"""Binomial pmf restated for the mean derivation.

Implements eq (4.60) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_60(k, n, p):
    """Binomial pmf restated for the mean derivation.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.60).
    """
    value = _morin.binomial_pmf(k, n, p)
    payload = {"probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Binomial pmf restated for the mean derivation.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e60: Binomial pmf restated for the mean derivation. Morin (2016) eq (4.60)."
