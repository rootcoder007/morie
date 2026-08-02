"""Noise specification Z ~ (mu=0, sigma_z) in the model Y = mX + Z.

Implements eq (6.3) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3(m, sigma_x, sigma_z):
    """Noise specification Z ~ (mu=0, sigma_z) in the model Y = mX + Z.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.3).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"mu_z": 0.0, "sigma_z": float(sigma_z), "sigma_y": sigma_y, "r": r}
    lines = [("sigma_y", sigma_y), ("r", r)]
    return RichResult(
        title="Noise specification Z ~ (mu=0, sigma_z) in the model Y = mX + Z.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e3: Noise specification Z ~ (mu=0, sigma_z) in the model Y = mX + Z. Morin (2016) eq (6.3)."
