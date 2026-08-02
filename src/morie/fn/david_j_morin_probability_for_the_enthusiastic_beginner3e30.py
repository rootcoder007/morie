"""Variance of a sum of n independent variables: sum of variances.

Implements eq (3.30) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_30(variances):
    """Variance of a sum of n independent variables: sum of variances.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.30).
    """
    value = _morin.var_sum_independent(variances)
    payload = {"var_sum": value}
    lines = [("Var(sum)", value)]
    return RichResult(
        title="Variance of a sum of n independent variables: sum of variances.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e30: Variance of a sum of n independent variables: sum of variances. Morin (2016) eq (3.30)."
