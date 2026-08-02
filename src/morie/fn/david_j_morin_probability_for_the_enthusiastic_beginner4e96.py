"""Binomial peak value PB(pn) at k = pn.

Implements eq (4.96) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_96(n, p):
    """Binomial peak value PB(pn) at k = pn.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.96).
    """
    k = int(round(float(p) * int(n)))
    value = _morin.binomial_pmf(k, n, p)
    payload = {"k": k, "PB": value}
    lines = [("PB(pn)", value)]
    return RichResult(
        title="Binomial peak value PB(pn) at k = pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e96: Binomial peak value PB(pn) at k = pn. Morin (2016) eq (4.96)."
