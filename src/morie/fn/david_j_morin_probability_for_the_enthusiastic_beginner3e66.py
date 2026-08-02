"""Computational identity: (1/n) sum (xi - xbar)^2 = mean(x^2) - xbar^2.

Implements eq (3.66) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_66(x):
    """Computational identity: (1/n) sum (xi - xbar)^2 = mean(x^2) - xbar^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.66).
    """
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    lhs = _morin.population_variance(x_a)
    rhs = float(np.mean(x_a ** 2)) - float(np.mean(x_a)) ** 2
    payload = {"lhs": lhs, "rhs": rhs, "identity_error": abs(lhs - rhs)}
    lines = [("both sides", lhs)]
    return RichResult(
        title="Computational identity: (1/n) sum (xi - xbar)^2 = mean(x^2) - xbar^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e66: Computational identity: (1/n) sum (xi - xbar)^2 = mean(x^2) - xbar^2. Morin (2016) eq (3.66)."
