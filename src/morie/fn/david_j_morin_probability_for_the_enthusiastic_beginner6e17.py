"""Convert model parameters (m, sigma_x, sigma_z) to (sigma_y, r).

Implements eq (6.17) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_17(m, sigma_x, sigma_z):
    """Convert model parameters (m, sigma_x, sigma_z) to (sigma_y, r).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.17).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"sigma_y": sigma_y, "r": r}
    lines = [("sigma_y", sigma_y), ("r", r)]
    return RichResult(
        title="Convert model parameters (m, sigma_x, sigma_z) to (sigma_y, r).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e17: Convert model parameters (m, sigma_x, sigma_z) to (sigma_y, r). Morin (2016) eq (6.17)."
