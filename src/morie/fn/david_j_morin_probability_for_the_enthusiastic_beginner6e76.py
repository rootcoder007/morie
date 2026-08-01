"""Worked spread: sigma_y = sqrt((1)^2 (7.5)^2 + (10.6)^2) = 13.

Implements eq (6.76) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76(m=1.0, sigma_x=7.5, sigma_z=10.6):
    """Worked spread: sigma_y = sqrt((1)^2 (7.5)^2 + (10.6)^2) = 13.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.76).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"sigma_y": sigma_y, "r": r}
    lines = [("sigma_y", sigma_y)]
    return RichResult(
        title="Worked spread: sigma_y = sqrt((1)^2 (7.5)^2 + (10.6)^2) = 13.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e76: Worked spread: sigma_y = sqrt((1)^2 (7.5)^2 + (10.6)^2) = 13. Morin (2016) eq (6.76)."
