"""Standard deviation of a discrete pmf (the OCR dropped the sqrt).

Implements eq (5.31) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_31(values, probs):
    """Standard deviation of a discrete pmf (the OCR dropped the sqrt).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.31).
    """
    sd, mu = _morin.pmf_sd(values, probs)
    payload = {"sd": sd, "mean": mu, "variance": sd * sd}
    lines = [("mean", mu), ("sd", sd)]
    return RichResult(
        title="Standard deviation of a discrete pmf (the OCR dropped the sqrt).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e31: Standard deviation of a discrete pmf (the OCR dropped the sqrt). Morin (2016) eq (5.31)."
