"""P(0) = e^(-a) as the alternating exponential series.

Implements eq (4.53) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_53"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_53(a, terms=60):
    """P(0) = e^(-a) as the alternating exponential series.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.53).
    """
    partials, closed = _morin.poisson_zero_series(a, terms)
    payload = {"partial_sums": partials, "e_minus_a": closed,
               "final_error": abs(partials[-1] - closed)}
    lines = [("series", partials[-1]), ("e^-a", closed)]
    return RichResult(
        title="P(0) = e^(-a) as the alternating exponential series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e53: P(0) = e^(-a) as the alternating exponential series. Morin (2016) eq (4.53)."
