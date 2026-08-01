"""Poisson-Stirling in centered variables k = a + x.

Implements eq (5.17) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17(x_dev, a):
    """Poisson-Stirling in centered variables k = a + x.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.17).
    """
    k = int(round(float(a) + float(x_dev)))
    if k < 1:
        raise ValueError("a + x must round to k >= 1")
    approx = _morin.poisson_stirling(k, a)
    exact = _morin.poisson_pmf(k, a)
    payload = {"k": k, "approx": approx, "exact": exact}
    lines = [("PP(a+x)", approx)]
    return RichResult(
        title="Poisson-Stirling in centered variables k = a + x.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e17: Poisson-Stirling in centered variables k = a + x. Morin (2016) eq (5.17)."
