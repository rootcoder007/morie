"""Model slope from data: m = Cov(x,y)/s_x^2.

Implements eq (6.13) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["slope_from_cov"]


def slope_from_cov(x, y):
    """Model slope from data: m = Cov(x,y)/s_x^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.13).
    """
    value = _morin.slope_from_cov(x, y)
    payload = {"slope": value}
    lines = [("m", value)]
    return RichResult(
        title="Model slope from data: m = Cov(x,y)/s_x^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e13: Model slope from data: m = Cov(x,y)/s_x^2. Morin (2016) eq (6.13)."
