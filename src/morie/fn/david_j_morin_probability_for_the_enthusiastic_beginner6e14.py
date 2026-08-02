"""Covariance shortcut: Cov = E(XY) - mu_x mu_y.

Implements eq (6.14) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, y):
    """Covariance shortcut: Cov = E(XY) - mu_x mu_y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.14).
    """
    value = _morin.cov_shortcut(x, y)
    payload = {"cov": value}
    lines = [("Cov(x, y)", value)]
    return RichResult(
        title="Covariance shortcut: Cov = E(XY) - mu_x mu_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e14: Covariance shortcut: Cov = E(XY) - mu_x mu_y. Morin (2016) eq (6.14)."
