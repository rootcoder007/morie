"""Mean of Y = mX + Z: mu_y = m mu_x + mu_z.

Implements eq (6.4) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4(m, mu_x, mu_z):
    """Mean of Y = mX + Z: mu_y = m mu_x + mu_z.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.4).
    """
    mu_y = float(m) * float(mu_x) + float(mu_z)
    payload = {"mu_y": mu_y}
    lines = [("mu_y", mu_y)]
    return RichResult(
        title="Mean of Y = mX + Z: mu_y = m mu_x + mu_z.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e4: Mean of Y = mX + Z: mu_y = m mu_x + mu_z. Morin (2016) eq (6.4)."
