"""sigma of a sum of n independent variables: sqrt(sum sigma_i^2).

Implements eq (3.43) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43(sigmas):
    """sigma of a sum of n independent variables: sqrt(sum sigma_i^2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.43).
    """
    value = _morin.sd_sum_independent(sigmas)
    payload = {"sd_sum": value}
    lines = [("sigma_sum", value)]
    return RichResult(
        title="sigma of a sum of n independent variables: sqrt(sum sigma_i^2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e43: sigma of a sum of n independent variables: sqrt(sum sigma_i^2). Morin (2016) eq (3.43)."
