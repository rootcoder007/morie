"""Law of total probability across a partition (three-seat worked example).

Implements eq (2.29) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29(priors, likelihoods):
    """Law of total probability across a partition (three-seat worked example).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.29).
    """
    value = _morin.total_probability(priors, likelihoods)
    payload = {"p_event": value}
    lines = [("P(event)", value)]
    return RichResult(
        title="Law of total probability across a partition (three-seat worked example).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e29: Law of total probability across a partition (three-seat worked example). Morin (2016) eq (2.29)."
