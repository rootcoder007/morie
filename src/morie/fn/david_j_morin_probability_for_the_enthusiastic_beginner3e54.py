"""Sample mean X-bar = (X1 + ... + Xn)/n.

Implements eq (3.54) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_54(x):
    """Sample mean X-bar = (X1 + ... + Xn)/n.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.54).
    """
    value = _morin.sample_mean(x)
    payload = {"mean": value, "n": int(np.atleast_1d(x).size)}
    lines = [("X-bar", value)]
    return RichResult(
        title="Sample mean X-bar = (X1 + ... + Xn)/n.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e54: Sample mean X-bar = (X1 + ... + Xn)/n. Morin (2016) eq (3.54)."
