"""Poisson variance: E(k^2) - a^2 = a.

Implements eq (4.94) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94(a):
    """Poisson variance: E(k^2) - a^2 = a.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.94).
    """
    mean, var = _morin.poisson_mean_var(a)
    payload = {"variance": var}
    lines = [("variance", var)]
    return RichResult(
        title="Poisson variance: E(k^2) - a^2 = a.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e94: Poisson variance: E(k^2) - a^2 = a. Morin (2016) eq (4.94)."
