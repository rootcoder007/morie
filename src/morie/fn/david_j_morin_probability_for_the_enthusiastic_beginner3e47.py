"""sigma of the number of Heads in n biased flips: sqrt(npq).

Implements eq (3.47) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_47(n, p):
    """sigma of the number of Heads in n biased flips: sqrt(npq).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.47).
    """
    value = _morin.sd_binomial(n, p)
    payload = {"n": int(n), "p": float(p), "sd": value}
    lines = [("sqrt(npq)", value)]
    return RichResult(
        title="sigma of the number of Heads in n biased flips: sqrt(npq).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e47: sigma of the number of Heads in n biased flips: sqrt(npq). Morin (2016) eq (3.47)."
