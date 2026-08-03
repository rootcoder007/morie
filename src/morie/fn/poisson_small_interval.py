"""Tiny-interval Poisson probability: P(1) ~ lambda eps.

Implements eq (4.18) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["poisson_small_interval"]


def poisson_small_interval(lam, eps):
    """Tiny-interval Poisson probability: P(1) ~ lambda eps.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.18).
    """
    approx, exact = _morin.poisson_small_interval(lam, eps)
    payload = {"approx": approx, "exact": exact,
               "abs_error": abs(approx - exact)}
    lines = [("lambda*eps", approx), ("exact P(1)", exact)]
    return RichResult(
        title="Tiny-interval Poisson probability: P(1) ~ lambda eps.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e18: Tiny-interval Poisson probability: P(1) ~ lambda eps. Morin (2016) eq (4.18)."
