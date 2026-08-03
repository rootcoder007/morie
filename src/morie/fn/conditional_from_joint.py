"""Conditional probability from the joint: P(B|A) = P(A and B)/P(A).

Implements eq (2.48) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["conditional_from_joint"]


def conditional_from_joint(p_a_and_b, p_a):
    """Conditional probability from the joint: P(B|A) = P(A and B)/P(A).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.48).
    """
    value = _morin.conditional_from_joint(p_a_and_b, p_a)
    payload = {"p_a_and_b": float(p_a_and_b), "p_a": float(p_a),
               "p_b_given_a": value}
    lines = [("P(B|A)", value)]
    return RichResult(
        title="Conditional probability from the joint: P(B|A) = P(A and B)/P(A).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e48: Conditional probability from the joint: P(B|A) = P(A and B)/P(A). Morin (2016) eq (2.48)."
