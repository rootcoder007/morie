"""Factorial N! via the book's product definition.

Implements eq (1.1) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_1"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_1(n):
    """Factorial N! via the book's product definition.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.1).
    """
    value = _morin.factorial(n)
    payload = {"n": int(n), "factorial": value}
    lines = [("n", int(n)), ("n!", value)]
    return RichResult(
        title="Factorial N! via the book's product definition.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e1: Factorial N! via the book's product definition. Morin (2016) eq (1.1)."
