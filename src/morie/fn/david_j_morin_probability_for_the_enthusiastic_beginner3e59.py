"""Variance definition restated (summary): E[(X-mu)^2].

Implements eq (3.59) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59(values, probs):
    """Variance definition restated (summary): E[(X-mu)^2].

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.59).
    """
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu}
    lines = [("variance", variance)]
    return RichResult(
        title="Variance definition restated (summary): E[(X-mu)^2].",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e59: Variance definition restated (summary): E[(X-mu)^2]. Morin (2016) eq (3.59)."
