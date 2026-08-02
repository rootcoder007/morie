"""Independence makes the covariance vanish: Cov = E(X)E(Y) - mu_x mu_y = 0.

Implements eq (6.63) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63(x, y, tol=1e-9):
    """Independence makes the covariance vanish: Cov = E(X)E(Y) - mu_x mu_y = 0.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.63).
    """
    cov = _morin.cov_shortcut(x, y)
    payload = {"cov": cov, "near_zero": abs(cov) <= float(tol)}
    lines = [("Cov", cov)]
    return RichResult(
        title="Independence makes the covariance vanish: Cov = E(X)E(Y) - mu_x mu_y = 0.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e63: Independence makes the covariance vanish: Cov = E(X)E(Y) - mu_x mu_y = 0. Morin (2016) eq (6.63)."
