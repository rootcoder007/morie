"""General or rule: P(A or B) = P(A) + P(B) - P(A and B).

Implements eq (2.21) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["prob_or_general"]


def prob_or_general(p_a, p_b, p_ab):
    """General or rule: P(A or B) = P(A) + P(B) - P(A and B).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.21).
    """
    value = _morin.prob_or_general(p_a, p_b, p_ab)
    payload = {"p_a": float(p_a), "p_b": float(p_b), "p_ab": float(p_ab),
               "p_or": value}
    lines = [("P(A or B)", value)]
    return RichResult(
        title="General or rule: P(A or B) = P(A) + P(B) - P(A and B).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e21: General or rule: P(A or B) = P(A) + P(B) - P(A and B). Morin (2016) eq (2.21)."
