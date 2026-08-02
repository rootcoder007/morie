"""Poisson mean: sum k P(k) = a.

Implements eq (4.92) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92(a):
    """Poisson mean: sum k P(k) = a.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.92).
    """
    mean, var = _morin.poisson_mean_var(a)
    payload = {"mean": mean}
    lines = [("sum k P(k)", mean)]
    return RichResult(
        title="Poisson mean: sum k P(k) = a.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e92: Poisson mean: sum k P(k) = a. Morin (2016) eq (4.92)."
