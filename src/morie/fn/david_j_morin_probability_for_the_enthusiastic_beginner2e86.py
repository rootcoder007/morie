"""Decompose P(B) over A and not-A.

Implements eq (2.86) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86(p_a, p_b_given_a, p_b_given_not_a):
    """Decompose P(B) over A and not-A.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.86).
    """
    p_a_f = float(p_a)
    value = _morin.total_probability([p_a_f, 1.0 - p_a_f],
                                     [p_b_given_a, p_b_given_not_a])
    payload = {"p_b": value}
    lines = [("P(B)", value)]
    return RichResult(
        title="Decompose P(B) over A and not-A.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e86: Decompose P(B) over A and not-A. Morin (2016) eq (2.86)."
