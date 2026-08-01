"""Product rule for k independent events (two-dice worked example).

Implements eq (2.3) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3(ps):
    """Product rule for k independent events (two-dice worked example).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.3).
    """
    value = _morin.prob_and_independent(ps)
    payload = {"ps": [float(x) for x in np.atleast_1d(ps)], "p_and": value}
    lines = [("P(all events)", value)]
    return RichResult(
        title="Product rule for k independent events (two-dice worked example).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e3: Product rule for k independent events (two-dice worked example). Morin (2016) eq (2.3)."
