"""Expected-count Gaussian for N repetitions of a dice-sum experiment.

Implements eq (5.28) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28(x, n_reps=100000, mu=35.0, sigma=5.4):
    """Expected-count Gaussian for N repetitions of a dice-sum experiment.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.28).
    """
    n_r = float(n_reps)
    if n_r <= 0:
        raise ValueError("n_reps must be > 0")
    value = n_r * _morin.normal_pdf(x, mu, sigma)
    payload = {"expected_count": value, "mu": float(mu), "sigma": float(sigma)}
    lines = [("expected count", value)]
    return RichResult(
        title="Expected-count Gaussian for N repetitions of a dice-sum experiment.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e28: Expected-count Gaussian for N repetitions of a dice-sum experiment. Morin (2016) eq (5.28)."
