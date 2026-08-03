"""sigma of the average Heads fraction in n fair flips: 1/(2 sqrt(n)).

Implements eq (3.52) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["sd_fair_coin_avg"]


def sd_fair_coin_avg(n):
    """sigma of the average Heads fraction in n fair flips: 1/(2 sqrt(n)).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.52).
    """
    value = _morin.sd_fair_coin_avg(n)
    payload = {"n": int(n), "sd_avg": value}
    lines = [("sigma_avg", value)]
    return RichResult(
        title="sigma of the average Heads fraction in n fair flips: 1/(2 sqrt(n)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e52: sigma of the average Heads fraction in n fair flips: 1/(2 sqrt(n)). Morin (2016) eq (3.52)."
