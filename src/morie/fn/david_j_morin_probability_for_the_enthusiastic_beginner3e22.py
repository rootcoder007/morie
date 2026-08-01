"""Bernoulli variance p(1-p) = pq.

Implements eq (3.22) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_22"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_22(p):
    """Bernoulli variance p(1-p) = pq.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.22).
    """
    value = _morin.bernoulli_variance(p)
    payload = {"p": float(p), "variance": value}
    lines = [("pq", value)]
    return RichResult(
        title="Bernoulli variance p(1-p) = pq.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e22: Bernoulli variance p(1-p) = pq. Morin (2016) eq (3.22)."
