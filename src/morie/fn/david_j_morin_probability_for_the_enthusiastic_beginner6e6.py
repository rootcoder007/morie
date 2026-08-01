"""Correlation coefficient of the model: r = m sigma_x / sigma_y.

Implements eq (6.6) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_6(m, sigma_x, sigma_z):
    """Correlation coefficient of the model: r = m sigma_x / sigma_y.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.6).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"r": r, "sigma_y": sigma_y}
    lines = [("r", r)]
    return RichResult(
        title="Correlation coefficient of the model: r = m sigma_x / sigma_y.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e6: Correlation coefficient of the model: r = m sigma_x / sigma_y. Morin (2016) eq (6.6)."
