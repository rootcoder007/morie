"""Variance of the sum of two fair coin flips is 1/2.

Implements eq (3.28) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28():
    """Variance of the sum of two fair coin flips is 1/2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.28).
    """
    values, probs = _morin.pmf_sum_convolution([0.0, 1.0], [0.5, 0.5],
                                               [0.0, 1.0], [0.5, 0.5])
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu}
    lines = [("Var(X+Y)", variance)]
    return RichResult(
        title="Variance of the sum of two fair coin flips is 1/2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e28: Variance of the sum of two fair coin flips is 1/2. Morin (2016) eq (3.28)."
