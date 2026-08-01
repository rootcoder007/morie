"""Exponential second moment E(T^2) = 2 tau^2 and Var = tau^2.

Implements eq (4.85) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_85(tau):
    """Exponential second moment E(T^2) = 2 tau^2 and Var = tau^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.85).
    """
    mean, second, var = _morin.exponential_moments(tau)
    payload = {"second_moment": second, "variance": var}
    lines = [("E(T^2)", second), ("Var(T)", var)]
    return RichResult(
        title="Exponential second moment E(T^2) = 2 tau^2 and Var = tau^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e85: Exponential second moment E(T^2) = 2 tau^2 and Var = tau^2. Morin (2016) eq (4.85)."
