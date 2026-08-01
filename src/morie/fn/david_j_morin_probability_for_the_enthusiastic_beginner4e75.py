"""Hypergeometric -> binomial as the population grows.

Implements eq (4.75) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_75(k, n, p, N):
    """Hypergeometric -> binomial as the population grows.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.75).
    """
    hyper, binom_p, err = _morin.hypergeometric_binomial_limit(k, n, p, N)
    payload = {"hypergeometric": hyper, "binomial": binom_p, "abs_error": err}
    lines = [("abs error", err)]
    return RichResult(
        title="Hypergeometric -> binomial as the population grows.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e75: Hypergeometric -> binomial as the population grows. Morin (2016) eq (4.75)."
