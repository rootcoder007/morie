"""Unbiased sample variance s^2 with the n-1 denominator.

Implements eq (3.73) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_73(x):
    """Unbiased sample variance s^2 with the n-1 denominator.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.73).
    """
    value = _morin.sample_variance(x)
    payload = {"sample_variance": value,
               "population_variance": _morin.population_variance(x)}
    lines = [("s^2", value)]
    return RichResult(
        title="Unbiased sample variance s^2 with the n-1 denominator.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e73: Unbiased sample variance s^2 with the n-1 denominator. Morin (2016) eq (3.73)."
