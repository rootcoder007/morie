"""Hand uses at most two suits (inclusion-exclusion over suit pairs).

Implements eq (2.43) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_43"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_43(n_suits=4, n_ranks=13, hand=5):
    """Hand uses at most two suits (inclusion-exclusion over suit pairs).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.43).
    """
    favorable, total, prob = _morin.at_most_two_suits_probability(
        n_suits, n_ranks, hand)
    payload = {"favorable": favorable, "total": total, "probability": prob}
    lines = [("favorable hands", favorable), ("total hands", total),
             ("probability", prob)]
    return RichResult(
        title="Hand uses at most two suits (inclusion-exclusion over suit pairs).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e43: Hand uses at most two suits (inclusion-exclusion over suit pairs). Morin (2016) eq (2.43)."
