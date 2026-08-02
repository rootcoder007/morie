"""E of a sum of n i.i.d. variables: n E(X).

Implements eq (3.15) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15(e_x, n):
    """E of a sum of n i.i.d. variables: n E(X).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.15).
    """
    n_i = int(n)
    if n_i < 0 or n_i != float(n):
        raise ValueError("n must be a non-negative integer")
    value = n_i * float(e_x)
    payload = {"e_sum": value, "n": n_i}
    lines = [("E(X1 + ... + Xn)", value)]
    return RichResult(
        title="E of a sum of n i.i.d. variables: n E(X).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e15: E of a sum of n i.i.d. variables: n E(X). Morin (2016) eq (3.15)."
