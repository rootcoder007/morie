"""Poisson mode: P(k) maximal at k = ceil(a) - 1.

Implements eq (4.89) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_89(a):
    """Poisson mode: P(k) maximal at k = ceil(a) - 1.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.89).
    """
    k_star = _morin.poisson_mode(a)
    p_star = _morin.poisson_pmf(k_star, a)
    neighbors = {k_star - 1: (_morin.poisson_pmf(k_star - 1, a)
                              if k_star >= 1 else 0.0),
                 k_star + 1: _morin.poisson_pmf(k_star + 1, a)}
    if any(v > p_star + 1e-15 for v in neighbors.values()):
        raise AssertionError("neighbor beats the claimed mode")
    payload = {"mode": k_star, "p_mode": p_star}
    lines = [("mode k", k_star), ("P(mode)", p_star)]
    return RichResult(
        title="Poisson mode: P(k) maximal at k = ceil(a) - 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e89: Poisson mode: P(k) maximal at k = ceil(a) - 1. Morin (2016) eq (4.89)."
