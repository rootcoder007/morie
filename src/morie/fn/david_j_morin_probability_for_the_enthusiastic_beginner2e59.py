"""Bayes setup A/~A/Z mapping with explicit-form posterior.

Implements eq (2.59) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59(p_a, p_z_given_a, p_z_given_not_a):
    """Bayes setup A/~A/Z mapping with explicit-form posterior.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.59).
    """
    value = _morin.bayes_explicit(p_a, p_z_given_a, p_z_given_not_a)
    payload = {"posterior": value,
               "setup": {"A": "hypothesis", "not_A": "complement",
                         "Z": "observed evidence"}}
    lines = [("P(A|Z)", value)]
    return RichResult(
        title="Bayes setup A/~A/Z mapping with explicit-form posterior.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e59: Bayes setup A/~A/Z mapping with explicit-form posterior. Morin (2016) eq (2.59)."
