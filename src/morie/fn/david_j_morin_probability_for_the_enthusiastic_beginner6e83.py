"""Worked means of the five-point dataset: xbar = 4, ybar = 3.

Implements eq (6.83) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_83"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_83(x=None, y=None):
    """Worked means of the five-point dataset: xbar = 4, ybar = 3.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.83).
    """
    x_d = [2.0, 3.0, 3.0, 5.0, 7.0] if x is None else x
    y_d = [1.0, 1.0, 3.0, 4.0, 6.0] if y is None else y
    x_a = np.atleast_1d(np.asarray(x_d, dtype=float))
    y_a = np.atleast_1d(np.asarray(y_d, dtype=float))
    payload = {"xbar": float(x_a.mean()), "ybar": float(y_a.mean())}
    lines = [("xbar", float(x_a.mean())), ("ybar", float(y_a.mean()))]
    return RichResult(
        title="Worked means of the five-point dataset: xbar = 4, ybar = 3.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e83: Worked means of the five-point dataset: xbar = 4, ybar = 3. Morin (2016) eq (6.83)."
