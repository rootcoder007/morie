"""Var(aX) = a^2 Var(X).

Implements eq (3.24) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["var_scale"]


def var_scale(a, var_x):
    """Var(aX) = a^2 Var(X).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.24).
    """
    value = _morin.var_scale(a, var_x)
    payload = {"a": float(a), "var_x": float(var_x), "var_aX": value}
    lines = [("Var(aX)", value)]
    return RichResult(
        title="Var(aX) = a^2 Var(X).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e24: Var(aX) = a^2 Var(X). Morin (2016) eq (3.24)."
