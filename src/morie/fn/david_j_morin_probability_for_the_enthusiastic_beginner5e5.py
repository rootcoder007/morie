"""Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).

Implements eq (5.5) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5(x, n):
    """Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.5).
    """
    x_i, n_i = int(x), int(n)
    if abs(x_i) > n_i:
        value = 0.0
    else:
        value = (math.factorial(2 * n_i)
                 / (math.factorial(n_i + x_i) * math.factorial(n_i - x_i))
                 / 4.0 ** n_i)
    check = _morin.binomial_centered_pmf(x_i, n_i)
    if abs(value - check) > 1e-12 * max(1.0, check):
        raise AssertionError("factorial form disagrees with C(2n, n+x)/2^2n")
    payload = {"probability": value}
    lines = [("PB(x)", value)]
    return RichResult(
        title="Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e5: Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)). Morin (2016) eq (5.5)."
