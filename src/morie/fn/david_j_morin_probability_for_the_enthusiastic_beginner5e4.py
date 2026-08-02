"""Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) vs exact.

Implements eq (5.4) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_4(x, n):
    """Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) vs exact.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.4).
    """
    approx = _morin.gaussian_approx_2n(x, n)
    exact = _morin.binomial_centered_pmf(int(round(float(x))), n)
    payload = {"approx": approx, "exact": exact,
               "rel_error": abs(approx - exact) / max(exact, 1e-300)}
    lines = [("PG(x)", approx), ("PB(x)", exact)]
    return RichResult(
        title="Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) vs exact.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e4: Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) vs exact. Morin (2016) eq (5.4)."
