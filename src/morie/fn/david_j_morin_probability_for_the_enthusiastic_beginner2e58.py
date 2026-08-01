"""False-positive worked example via the explicit Bayes form.

Implements eq (2.58) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58(p_a=0.02, p_z_given_a=0.95, p_z_given_not_a=0.10):
    """False-positive worked example via the explicit Bayes form.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.58).
    """
    value = _morin.bayes_explicit(p_a, p_z_given_a, p_z_given_not_a)
    payload = {"posterior": value}
    lines = [("P(disease | positive)", value)]
    return RichResult(
        title="False-positive worked example via the explicit Bayes form.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e58: False-positive worked example via the explicit Bayes form. Morin (2016) eq (2.58)."
