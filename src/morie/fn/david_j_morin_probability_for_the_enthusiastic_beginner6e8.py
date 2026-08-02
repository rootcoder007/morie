"""Covariance of zero-mean variables: Cov(X,Y) = E(XY).

Implements eq (6.8) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8(x, y):
    """Covariance of zero-mean variables: Cov(X,Y) = E(XY).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.8).
    """
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    y_a = np.atleast_1d(np.asarray(y, dtype=float))
    if abs(float(x_a.mean())) > 1e-9 or abs(float(y_a.mean())) > 1e-9:
        raise ValueError("this form needs zero-mean data; use eq (6.14) otherwise")
    value = float(np.mean(x_a * y_a))
    payload = {"cov": value}
    lines = [("E(XY)", value)]
    return RichResult(
        title="Covariance of zero-mean variables: Cov(X,Y) = E(XY).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e8: Covariance of zero-mean variables: Cov(X,Y) = E(XY). Morin (2016) eq (6.8)."
