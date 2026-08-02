"""Binomial mean E(k) = np, verified against the pmf series.

Implements eq (4.61) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_61(n, p):
    """Binomial mean E(k) = np, verified against the pmf series.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.61).
    """
    value = _morin.binomial_mean(n, p)
    pmf = _morin.binomial_pmf_vector(n, p)
    series = float(np.sum(np.arange(int(n) + 1) * pmf))
    if abs(series - value) > 1e-9 * max(1.0, value):
        raise AssertionError("series mean disagrees with np")
    payload = {"mean": value, "series_mean": series}
    lines = [("np", value)]
    return RichResult(
        title="Binomial mean E(k) = np, verified against the pmf series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e61: Binomial mean E(k) = np, verified against the pmf series. Morin (2016) eq (4.61)."
