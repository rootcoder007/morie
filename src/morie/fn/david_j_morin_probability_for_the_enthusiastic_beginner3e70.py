"""E[X^2] = sigma^2 + mu^2.

Implements eq (3.70) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_70"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_70(sigma, mu):
    """E[X^2] = sigma^2 + mu^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.70).
    """
    value = _morin.e_x_squared(sigma, mu)
    payload = {"e_x2": value}
    lines = [("E[X^2]", value)]
    return RichResult(
        title="E[X^2] = sigma^2 + mu^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e70: E[X^2] = sigma^2 + mu^2. Morin (2016) eq (3.70)."
