"""Exponential mean: integral of t e^(-t/tau)/tau dt = tau.

Implements eq (4.83) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_83(tau):
    """Exponential mean: integral of t e^(-t/tau)/tau dt = tau.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.83).
    """
    mean, second, var = _morin.exponential_moments(tau)
    payload = {"mean": mean}
    lines = [("E(T)", mean)]
    return RichResult(
        title="Exponential mean: integral of t e^(-t/tau)/tau dt = tau.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e83: Exponential mean: integral of t e^(-t/tau)/tau dt = tau. Morin (2016) eq (4.83)."
