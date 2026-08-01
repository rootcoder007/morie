"""Variance of a fair die roll (book's worked value 2.92).

Implements eq (3.20) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_20(sides=6):
    """Variance of a fair die roll (book's worked value 2.92).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.20).
    """
    k = int(sides)
    if k < 1:
        raise ValueError("sides must be >= 1")
    values = np.arange(1, k + 1, dtype=float)
    probs = np.full(k, 1.0 / k)
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu, "sides": k}
    lines = [("mean", mu), ("variance", variance)]
    return RichResult(
        title="Variance of a fair die roll (book's worked value 2.92).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e20: Variance of a fair die roll (book's worked value 2.92). Morin (2016) eq (3.20)."
