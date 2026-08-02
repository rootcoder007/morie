"""Population variance s-tilde^2 of a data set.

Implements eq (3.60) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_60(x):
    """Population variance s-tilde^2 of a data set.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.60).
    """
    value = _morin.population_variance(x)
    payload = {"variance": value}
    lines = [("s-tilde^2", value)]
    return RichResult(
        title="Population variance s-tilde^2 of a data set.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e60: Population variance s-tilde^2 of a data set. Morin (2016) eq (3.60)."
