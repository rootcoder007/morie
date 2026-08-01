"""Suit full house: k of one suit plus the rest of another.

Implements eq (2.41) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_41"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_41(n_suits=4, n_ranks=13, k_major=3, k_minor=2):
    """Suit full house: k of one suit plus the rest of another.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.41).
    """
    favorable, total, prob = _morin.suit_full_house_probability(
        n_suits, n_ranks, k_major, k_minor)
    payload = {"favorable": favorable, "total": total, "probability": prob}
    lines = [("favorable hands", favorable), ("total hands", total),
             ("probability", prob)]
    return RichResult(
        title="Suit full house: k of one suit plus the rest of another.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e41: Suit full house: k of one suit plus the rest of another. Morin (2016) eq (2.41)."
