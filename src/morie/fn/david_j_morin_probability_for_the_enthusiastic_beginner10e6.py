"""Model spread sigma_y (module name from the literal 10.6 in eq (6.76)).

Implements eq (6.5; the 10.6 in eq (6.76)) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6(m=1.0, sigma_x=7.5, sigma_z=10.6):
    """Model spread sigma_y (module name from the literal 10.6 in eq (6.76)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.5; the 10.6 in eq (6.76)).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(m, 0.0, sigma_x, 0.0, sigma_z)
    payload = {"sigma_y": sigma_y, "r": r}
    lines = [("sigma_y", sigma_y)]
    return RichResult(
        title="Model spread sigma_y (module name from the literal 10.6 in eq (6.76)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner10e6: Model spread sigma_y (module name from the literal 10.6 in eq (6.76)). Morin (2016) eq (6.5; the 10.6 in eq (6.76))."
