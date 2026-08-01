"""Binomial second moment E(k^2) = p^2 n(n-1) + pn.

Implements eq (4.66) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66(n, p):
    """Binomial second moment E(k^2) = p^2 n(n-1) + pn.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.66).
    """
    value = _morin.binomial_second_moment(n, p)
    pmf = _morin.binomial_pmf_vector(n, p)
    ks = np.arange(int(n) + 1)
    series = float(np.sum(ks ** 2 * pmf))
    if abs(series - value) > 1e-9 * max(1.0, value):
        raise AssertionError("series second moment disagrees")
    payload = {"second_moment": value}
    lines = [("E(k^2)", value)]
    return RichResult(
        title="Binomial second moment E(k^2) = p^2 n(n-1) + pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e66: Binomial second moment E(k^2) = p^2 n(n-1) + pn. Morin (2016) eq (4.66)."
