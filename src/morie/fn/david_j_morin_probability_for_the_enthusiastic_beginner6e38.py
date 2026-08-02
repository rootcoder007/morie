"""IQ test-retest worked example: r = 1/sqrt(2) ~ 0.71.

Implements eq (6.38) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_38"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_38():
    """IQ test-retest worked example: r = 1/sqrt(2) ~ 0.71.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.38).
    """
    sigma_x = 15.0 / math.sqrt(2.0)
    mu_y, sigma_y, r = _morin.linear_model_stats(1.0, 0.0, sigma_x, 0.0, sigma_x)
    payload = {"r": r, "sigma_y": sigma_y}
    lines = [("r", r)]
    return RichResult(
        title="IQ test-retest worked example: r = 1/sqrt(2) ~ 0.71.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e38: IQ test-retest worked example: r = 1/sqrt(2) ~ 0.71. Morin (2016) eq (6.38)."
