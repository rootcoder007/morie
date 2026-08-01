"""Population correlation r = Cov(X,Y)/(sigma_x sigma_y).

Implements eq (6.53) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53(x, y):
    """Population correlation r = Cov(X,Y)/(sigma_x sigma_y).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.53).
    """
    value = _morin.sample_r(x, y)
    A, C, r = _morin.regression_slope_product(x, y)
    payload = {"r": value, "slope_product_AC": A * C}
    lines = [("r", value), ("A*C = r^2", A * C)]
    return RichResult(
        title="Population correlation r = Cov(X,Y)/(sigma_x sigma_y).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e53: Population correlation r = Cov(X,Y)/(sigma_x sigma_y). Morin (2016) eq (6.53)."
