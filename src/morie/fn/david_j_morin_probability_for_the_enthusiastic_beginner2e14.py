"""Or rule for exclusive events: P(A or B) = P(A) + P(B).

Implements eq (2.14) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_14(ps):
    """Or rule for exclusive events: P(A or B) = P(A) + P(B).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.14).
    """
    value = _morin.prob_or_exclusive(ps)
    payload = {"ps": [float(x) for x in np.atleast_1d(ps)], "p_or": value}
    lines = [("P(any event)", value)]
    return RichResult(
        title="Or rule for exclusive events: P(A or B) = P(A) + P(B).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e14: Or rule for exclusive events: P(A or B) = P(A) + P(B). Morin (2016) eq (2.14)."
