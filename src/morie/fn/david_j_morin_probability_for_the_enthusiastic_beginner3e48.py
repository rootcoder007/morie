"""sigma of the number of Heads in n fair flips: sqrt(n)/2.

Implements eq (3.48) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_48(n):
    """sigma of the number of Heads in n fair flips: sqrt(n)/2.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.48).
    """
    value = _morin.sd_fair_coin_sum(n)
    payload = {"n": int(n), "sd": value}
    lines = [("sqrt(n)/2", value)]
    return RichResult(
        title="sigma of the number of Heads in n fair flips: sqrt(n)/2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e48: sigma of the number of Heads in n fair flips: sqrt(n)/2. Morin (2016) eq (3.48)."
