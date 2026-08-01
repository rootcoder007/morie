"""Gaussian approximation for n fair flips: e^(-2x^2/n)/sqrt(pi n/2).

Implements eq (5.14) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_14(x, n):
    """Gaussian approximation for n fair flips: e^(-2x^2/n)/sqrt(pi n/2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.14).
    """
    value = _morin.gaussian_approx_n(x, n)
    payload = {"x": float(x), "n": int(n), "PG": value}
    lines = [("PG(x)", value)]
    return RichResult(
        title="Gaussian approximation for n fair flips: e^(-2x^2/n)/sqrt(pi n/2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e14: Gaussian approximation for n fair flips: e^(-2x^2/n)/sqrt(pi n/2). Morin (2016) eq (5.14)."
