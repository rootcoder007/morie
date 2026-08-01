"""Gaussian approximation PG(x) = e^(-x^2/n)/sqrt(pi n), 2n flips.

Implements eq (5.13) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_13"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_13(x, n):
    """Gaussian approximation PG(x) = e^(-x^2/n)/sqrt(pi n), 2n flips.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.13).
    """
    value = _morin.gaussian_approx_2n(x, n)
    payload = {"x": float(x), "n": int(n), "PG": value}
    lines = [("PG(x)", value)]
    return RichResult(
        title="Gaussian approximation PG(x) = e^(-x^2/n)/sqrt(pi n), 2n flips.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e13: Gaussian approximation PG(x) = e^(-x^2/n)/sqrt(pi n), 2n flips. Morin (2016) eq (5.13)."
