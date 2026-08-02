"""Series product check used for the Poisson normalization.

Implements eq (7.10) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_10"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_10(a, terms=40):
    """Series product check used for the Poisson normalization.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.10).
    """
    partials, closed = _morin.exp_taylor(a, terms)
    total = partials[-1] * math.exp(-float(a))
    payload = {"normalization": total, "error": abs(total - 1.0)}
    lines = [("sum P(k)", total)]
    return RichResult(
        title="Series product check used for the Poisson normalization.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e10: Series product check used for the Poisson normalization. Morin (2016) eq (7.10)."
