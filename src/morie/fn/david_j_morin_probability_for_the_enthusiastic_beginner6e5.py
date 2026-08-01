"""Spread of Y = mX + Z: sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2).

Implements eq (6.5) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_5(m, sigma_x, sigma_z):
    """Spread of Y = mX + Z: sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.5).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"sigma_y": sigma_y}
    lines = [("sigma_y", sigma_y)]
    return RichResult(
        title="Spread of Y = mX + Z: sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e5: Spread of Y = mX + Z: sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2). Morin (2016) eq (6.5)."
