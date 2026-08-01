"""Expanded evidence: P(Z) = P(Z|A)P(A) + P(Z|~A)P(~A).

Implements eq (2.55) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55(p_a, p_z_given_a, p_z_given_not_a):
    """Expanded evidence: P(Z) = P(Z|A)P(A) + P(Z|~A)P(~A).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.55).
    """
    p_a_f = float(p_a)
    value = _morin.total_probability([p_a_f, 1.0 - p_a_f],
                                     [p_z_given_a, p_z_given_not_a])
    payload = {"p_z": value}
    lines = [("P(Z)", value)]
    return RichResult(
        title="Expanded evidence: P(Z) = P(Z|A)P(A) + P(Z|~A)P(~A).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e55: Expanded evidence: P(Z) = P(Z|A)P(A) + P(Z|~A)P(~A). Morin (2016) eq (2.55)."
