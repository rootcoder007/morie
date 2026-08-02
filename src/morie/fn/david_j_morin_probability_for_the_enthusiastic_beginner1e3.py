"""P_N = N!: number of permutations of N distinct objects.

Implements eq (1.3) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_3(n):
    """P_N = N!: number of permutations of N distinct objects.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.3).
    """
    value = _morin.permutations_count(n)
    payload = {"n": int(n), "permutations": value}
    lines = [("N", int(n)), ("P_N = N!", value)]
    return RichResult(
        title="P_N = N!: number of permutations of N distinct objects.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e3: P_N = N!: number of permutations of N distinct objects. Morin (2016) eq (1.3)."
