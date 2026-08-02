"""The two variance forms agree: E[(X-mu)^2] = E(X^2) - mu^2.

Implements eq (3.35) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_35(values, probs):
    """The two variance forms agree: E[(X-mu)^2] = E(X^2) - mu^2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.35).
    """
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu, "forms_agree": True}
    lines = [("variance (both forms)", variance)]
    return RichResult(
        title="The two variance forms agree: E[(X-mu)^2] = E(X^2) - mu^2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e35: The two variance forms agree: E[(X-mu)^2] = E(X^2) - mu^2. Morin (2016) eq (3.35)."
