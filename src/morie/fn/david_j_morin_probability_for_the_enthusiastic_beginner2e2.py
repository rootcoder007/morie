"""And rule for two independent events: P(A and B) = P(A) P(B).

Implements eq (2.2) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_2(p_a, p_b):
    """And rule for two independent events: P(A and B) = P(A) P(B).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.2).
    """
    value = _morin.prob_and_independent([p_a, p_b])
    payload = {"p_a": float(p_a), "p_b": float(p_b), "p_and": value}
    lines = [("P(A and B)", value)]
    return RichResult(
        title="And rule for two independent events: P(A and B) = P(A) P(B).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e2: And rule for two independent events: P(A and B) = P(A) P(B). Morin (2016) eq (2.2)."
