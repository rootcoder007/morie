"""Test-retest with equal signal and noise: r = 1/sqrt(2).

Implements eq (6.37) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37(sigma_signal, sigma_noise):
    """Test-retest with equal signal and noise: r = 1/sqrt(2).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (6.37).
    """
    mu_y, sigma_y, r = _morin.linear_model_stats(1.0, 0.0, sigma_signal,
                                                 0.0, sigma_noise)
    payload = {"r": r}
    lines = [("r", r)]
    return RichResult(
        title="Test-retest with equal signal and noise: r = 1/sqrt(2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e37: Test-retest with equal signal and noise: r = 1/sqrt(2). Morin (2016) eq (6.37)."
