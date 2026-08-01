"""Convolution integral for the sum of independent variables.

Implements eq (6.65) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_65(grid_x, density_x, grid_y, density_y, z):
    """Convolution integral for the sum of independent variables.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.65).
    """
    value = _morin.sum_density_convolution(grid_x, density_x,
                                           grid_y, density_y, z)
    payload = {"z": float(z), "density": value}
    lines = [("rho_Z(z)", value)]
    return RichResult(
        title="Convolution integral for the sum of independent variables.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e65: Convolution integral for the sum of independent variables. Morin (2016) eq (6.65)."
