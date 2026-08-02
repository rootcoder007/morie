"""sigma_aX = |a| sigma_X.

Implements eq (3.41) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_41"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_41(a, sigma):
    """sigma_aX = |a| sigma_X.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.41).
    """
    value = _morin.sd_scale(a, sigma)
    payload = {"a": float(a), "sigma": float(sigma), "sd_aX": value}
    lines = [("sigma_aX", value)]
    return RichResult(
        title="sigma_aX = |a| sigma_X.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e41: sigma_aX = |a| sigma_X. Morin (2016) eq (3.41)."
