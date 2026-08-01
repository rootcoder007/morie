"""Partial permutations N_P_n = N(N-1)...(N-(n-1)) = N!/(N-n)!.

Implements eq (1.5) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_5(N, n):
    """Partial permutations N_P_n = N(N-1)...(N-(n-1)) = N!/(N-n)!.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.5).
    """
    value = _morin.partial_permutations(N, n)
    payload = {"N": int(N), "n": int(n), "partial_permutations": value}
    lines = [("N", int(N)), ("n", int(n)), ("N_P_n", value)]
    return RichResult(
        title="Partial permutations N_P_n = N(N-1)...(N-(n-1)) = N!/(N-n)!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e5: Partial permutations N_P_n = N(N-1)...(N-(n-1)) = N!/(N-n)!. Morin (2016) eq (1.5)."
