"""Rearranged binomial before the three Poisson approximations.

Implements eq (4.35) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_35(k, n, a):
    """Rearranged binomial before the three Poisson approximations.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.35).
    """
    exact, limit, err = _morin.binomial_poisson_limit(k, n, a)
    payload = {"binomial": exact, "poisson": limit, "abs_error": err}
    lines = [("binomial", exact), ("Poisson limit", limit)]
    return RichResult(
        title="Rearranged binomial before the three Poisson approximations.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e35: Rearranged binomial before the three Poisson approximations. Morin (2016) eq (4.35)."
