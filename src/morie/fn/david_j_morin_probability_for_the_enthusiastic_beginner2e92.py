"""Inclusion-exclusion for three events.

Implements eq (2.92) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_92(p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc):
    """Inclusion-exclusion for three events.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.92).
    """
    value = _morin.inclusion_exclusion_3(p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc)
    payload = {"p_or": value}
    lines = [("P(A or B or C)", value)]
    return RichResult(
        title="Inclusion-exclusion for three events.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e92: Inclusion-exclusion for three events. Morin (2016) eq (2.92)."
