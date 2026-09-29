"""Conditional probability when B is a subset of A: P(B|A) = P(B)/P(A).

Implements eq (2.49) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["conditional_subset"]


def conditional_subset(p_b, p_a):
    """Conditional probability when B is a subset of A: P(B|A) = P(B)/P(A).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.49).

    Examples
    --------
    >>> round(conditional_subset(0.1, 0.4)["p_b_given_a"], 12)
    0.25
    """
    value = _morin.conditional_subset(p_b, p_a)
    payload = {"p_b": float(p_b), "p_a": float(p_a), "p_b_given_a": value}
    lines = [("P(B|A)", value)]
    return RichResult(
        title="Conditional probability when B is a subset of A: P(B|A) = P(B)/P(A).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e49: Conditional probability when B is a subset of A: P(B|A) = P(B)/P(A). Morin (2016) eq (2.49)."
