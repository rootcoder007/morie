"""Both chain-rule factorizations of P(A and B) agree.

Implements eq (2.9) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_9(p_a, p_b_given_a, p_b, p_a_given_b):
    """Both chain-rule factorizations of P(A and B) agree.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.9).
    """
    first = _morin.chain_rule(p_a, p_b_given_a)
    second = _morin.chain_rule(p_b, p_a_given_b)
    if abs(first - second) > 1e-9:
        raise ValueError("the two factorizations disagree; inputs inconsistent")
    payload = {"via_a": first, "via_b": second, "p_and": first}
    lines = [("P(A)P(B|A)", first), ("P(B)P(A|B)", second)]
    return RichResult(
        title="Both chain-rule factorizations of P(A and B) agree.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e9: Both chain-rule factorizations of P(A and B) agree. Morin (2016) eq (2.9)."
