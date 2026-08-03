"""Peak ratio PP(pn)/PB(pn) -> sqrt(1-p).

Implements eq (4.98) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["poisson_binomial_peak_ratio"]


def poisson_binomial_peak_ratio(n, p):
    """Peak ratio PP(pn)/PB(pn) -> sqrt(1-p).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.98).
    """
    ratio, limit = _morin.poisson_binomial_peak_ratio(n, p)
    payload = {"ratio": ratio, "sqrt_1_minus_p": limit,
               "abs_error": abs(ratio - limit)}
    lines = [("ratio", ratio), ("sqrt(1-p)", limit)]
    return RichResult(
        title="Peak ratio PP(pn)/PB(pn) -> sqrt(1-p).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e98: Peak ratio PP(pn)/PB(pn) -> sqrt(1-p). Morin (2016) eq (4.98)."
