"""At least one of k i.i.d. independent events, inclusion-exclusion form.

Implements eq (2.96) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_96"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_96(p, k=3):
    """At least one of k i.i.d. independent events, inclusion-exclusion form.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.96).
    """
    value = _morin.at_least_one_of_iid(p, k)
    payload = {"p": float(p), "k": int(k), "p_at_least_one": value}
    lines = [("P(at least one)", value)]
    return RichResult(
        title="At least one of k i.i.d. independent events, inclusion-exclusion form.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e96: At least one of k i.i.d. independent events, inclusion-exclusion form. Morin (2016) eq (2.96)."
