"""Variance of the sample mean: sigma^2 / N.

Implements eq (3.92) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_92(sigma, N):
    """Variance of the sample mean: sigma^2 / N.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.92).
    """
    value = _morin.var_of_sample_mean(sigma, N)
    payload = {"var_mean": value}
    lines = [("sigma^2/N", value)]
    return RichResult(
        title="Variance of the sample mean: sigma^2 / N.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e92: Variance of the sample mean: sigma^2 / N. Morin (2016) eq (3.92)."
