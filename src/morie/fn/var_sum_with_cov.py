"""Var(X + Y) expansion with the cross term 2Cov(X, Y).

Implements eq (3.26) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["var_sum_with_cov"]


def var_sum_with_cov(var_x, var_y, cov_xy=0.0):
    """Var(X + Y) expansion with the cross term 2Cov(X, Y).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.26).
    """
    value = _morin.var_sum_with_cov(var_x, var_y, cov_xy)
    payload = {"var_sum": value, "cov_xy": float(cov_xy)}
    lines = [("Var(X+Y)", value)]
    return RichResult(
        title="Var(X + Y) expansion with the cross term 2Cov(X, Y).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e26: Var(X + Y) expansion with the cross term 2Cov(X, Y). Morin (2016) eq (3.26)."


# compact alias per ledger/NAMING.md
varsumwithcov = var_sum_with_cov
