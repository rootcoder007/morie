"""Expected event count in time t equals lambda t (series-checked).

Implements eq (4.19) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_19"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_19(lam, t):
    """Expected event count in time t equals lambda t (series-checked).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.19).
    """
    value = _morin.poisson_mean_rate(lam, t)
    payload = {"lambda": float(lam), "t": float(t), "expected_events": value}
    lines = [("lambda t", value)]
    return RichResult(
        title="Expected event count in time t equals lambda t (series-checked).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e19: Expected event count in time t equals lambda t (series-checked). Morin (2016) eq (4.19)."
