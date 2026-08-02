"""Classify an event pair: independent and/or exclusive.

Implements eq (2.24) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_24"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_24(p_a, p_b, p_ab, tol=1e-12):
    """Classify an event pair: independent and/or exclusive.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.24).
    """
    independent, exclusive = _morin.classify_events(p_a, p_b, p_ab, tol)
    payload = {"independent": independent, "exclusive": exclusive,
               "p_a": float(p_a), "p_b": float(p_b), "p_ab": float(p_ab)}
    lines = [("independent", independent), ("exclusive", exclusive)]
    return RichResult(
        title="Classify an event pair: independent and/or exclusive.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e24: Classify an event pair: independent and/or exclusive. Morin (2016) eq (2.24)."
