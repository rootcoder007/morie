"""Standard deviation sigma_X = sqrt(Var(X)).

Implements eq (3.39) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39(var_x):
    """Standard deviation sigma_X = sqrt(Var(X)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.39).
    """
    v = float(var_x)
    if v < 0:
        raise ValueError("variance must be >= 0")
    value = math.sqrt(v)
    payload = {"variance": v, "sd": value}
    lines = [("sigma", value)]
    return RichResult(
        title="Standard deviation sigma_X = sqrt(Var(X)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e39: Standard deviation sigma_X = sqrt(Var(X)). Morin (2016) eq (3.39)."
