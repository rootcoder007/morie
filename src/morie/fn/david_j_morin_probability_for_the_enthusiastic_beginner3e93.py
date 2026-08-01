"""sd of the sample mean sigma/sqrt(N) never exceeds sigma.

Implements eq (3.93) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_93(sigma, N):
    """sd of the sample mean sigma/sqrt(N) never exceeds sigma.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.93).
    """
    var_mean = _morin.var_of_sample_mean(sigma, N)
    value = math.sqrt(var_mean)
    if value > float(sigma) + 1e-12:
        raise AssertionError("sd of the mean exceeded sigma")
    payload = {"sd_mean": value, "sigma": float(sigma), "bounded": True}
    lines = [("sigma/sqrt(N)", value)]
    return RichResult(
        title="sd of the sample mean sigma/sqrt(N) never exceeds sigma.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e93: sd of the sample mean sigma/sqrt(N) never exceeds sigma. Morin (2016) eq (3.93)."
