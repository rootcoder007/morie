"""Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).

Implements eq (5.23) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_23(k, a):
    """Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.23).
    """
    value = _morin.poisson_gaussian(k, a)
    exact = _morin.poisson_pmf(int(round(float(k))), a) if float(k) >= 0 else 0.0
    payload = {"PG": value, "exact": exact}
    lines = [("PG(k)", value)]
    return RichResult(
        title="Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e23: Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a). Morin (2016) eq (5.23)."
