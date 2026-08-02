"""sigma = sqrt(E(X^2) - mu^2) for a discrete pmf.

Implements eq (3.40) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_40(values, probs):
    """sigma = sqrt(E(X^2) - mu^2) for a discrete pmf.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.40).
    """
    variance, mu = _morin.pmf_variance(values, probs)
    value = math.sqrt(variance)
    payload = {"sd": value, "variance": variance, "mean": mu}
    lines = [("sigma", value)]
    return RichResult(
        title="sigma = sqrt(E(X^2) - mu^2) for a discrete pmf.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e40: sigma = sqrt(E(X^2) - mu^2) for a discrete pmf. Morin (2016) eq (3.40)."
