"""Binomial with p = a/n against its Poisson limit.

Implements eq (4.34) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_34(k, n, a):
    """Binomial with p = a/n against its Poisson limit.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.34).
    """
    exact, limit, err = _morin.binomial_poisson_limit(k, n, a)
    payload = {"binomial": exact, "poisson": limit, "abs_error": err}
    lines = [("binomial", exact), ("Poisson limit", limit)]
    return RichResult(
        title="Binomial with p = a/n against its Poisson limit.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e34: Binomial with p = a/n against its Poisson limit. Morin (2016) eq (4.34)."
