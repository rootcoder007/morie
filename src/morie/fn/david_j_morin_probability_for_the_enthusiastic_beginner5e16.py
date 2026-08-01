"""Poisson pmf with Stirling's approximation applied to k!.

Implements eq (5.16) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_16(k, a):
    """Poisson pmf with Stirling's approximation applied to k!.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.16).
    """
    approx = _morin.poisson_stirling(k, a)
    exact = _morin.poisson_pmf(k, a)
    payload = {"approx": approx, "exact": exact,
               "rel_error": abs(approx - exact) / max(exact, 1e-300)}
    lines = [("Stirling PP(k)", approx), ("exact", exact)]
    return RichResult(
        title="Poisson pmf with Stirling's approximation applied to k!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e16: Poisson pmf with Stirling's approximation applied to k!. Morin (2016) eq (5.16)."
