"""Worked dice-total sigma: sqrt(n pq) with n=10,000, p=1/6 gives 37.

Implements eq (3.56) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_56(n=10000, p=1.0/6.0):
    """Worked dice-total sigma: sqrt(n pq) with n=10,000, p=1/6 gives 37.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.56).
    """
    value = _morin.sd_binomial(n, p)
    payload = {"n": int(n), "p": float(p), "sd_tot": value}
    lines = [("sigma_tot", value)]
    return RichResult(
        title="Worked dice-total sigma: sqrt(n pq) with n=10,000, p=1/6 gives 37.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e56: Worked dice-total sigma: sqrt(n pq) with n=10,000, p=1/6 gives 37. Morin (2016) eq (3.56)."
