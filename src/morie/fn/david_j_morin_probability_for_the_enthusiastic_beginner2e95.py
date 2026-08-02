"""Pairwise and triple intersections of i.i.d. independent events.

Implements eq (2.95) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95(p, k=2):
    """Pairwise and triple intersections of i.i.d. independent events.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.95).
    """
    p_f = float(p)
    if not 0.0 <= p_f <= 1.0:
        raise ValueError("p must be in [0, 1]")
    value = p_f ** int(k)
    payload = {"p": p_f, "k": int(k), "p_intersection": value}
    lines = [("P(all k events)", value)]
    return RichResult(
        title="Pairwise and triple intersections of i.i.d. independent events.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e95: Pairwise and triple intersections of i.i.d. independent events. Morin (2016) eq (2.95)."
