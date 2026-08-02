"""Variance of a set S of numbers: (1/n) sum (xi - xbar)^2.

Implements eq (3.37) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_37(x):
    """Variance of a set S of numbers: (1/n) sum (xi - xbar)^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.37).
    """
    value = _morin.population_variance(x)
    payload = {"variance": value, "n": int(np.atleast_1d(x).size)}
    lines = [("s-tilde^2", value)]
    return RichResult(
        title="Variance of a set S of numbers: (1/n) sum (xi - xbar)^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e37: Variance of a set S of numbers: (1/n) sum (xi - xbar)^2. Morin (2016) eq (3.37)."
