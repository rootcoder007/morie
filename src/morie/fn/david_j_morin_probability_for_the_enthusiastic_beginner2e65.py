"""Exact P(n Heads in 2n fair flips) = C(2n,n)/4^n.

Implements eq (2.65) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_65"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_65(n):
    """Exact P(n Heads in 2n fair flips) = C(2n,n)/4^n.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.65).
    """
    value = _morin.exact_half_heads(n)
    payload = {"n": int(n), "probability": value}
    lines = [("P(exactly n of 2n)", value)]
    return RichResult(
        title="Exact P(n Heads in 2n fair flips) = C(2n,n)/4^n.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e65: Exact P(n Heads in 2n fair flips) = C(2n,n)/4^n. Morin (2016) eq (2.65)."
