"""Gaussian approximation for n biased flips, centered at pn.

Implements eq (5.15) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_15"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_15(x, n, p):
    """Gaussian approximation for n biased flips, centered at pn.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.15).
    """
    value = _morin.gaussian_approx_biased(x, n, p)
    payload = {"x": float(x), "n": int(n), "p": float(p), "PG": value}
    lines = [("PG(x)", value)]
    return RichResult(
        title="Gaussian approximation for n biased flips, centered at pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e15: Gaussian approximation for n biased flips, centered at pn. Morin (2016) eq (5.15)."
