"""sigma_tot = sqrt(n)/2 for n fair coins (worked variant).

Implements eq (3.51) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51(n):
    """sigma_tot = sqrt(n)/2 for n fair coins (worked variant).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.51).
    """
    value = _morin.sd_fair_coin_sum(n)
    payload = {"n": int(n), "sd_tot": value}
    lines = [("sigma_tot", value)]
    return RichResult(
        title="sigma_tot = sqrt(n)/2 for n fair coins (worked variant).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e51: sigma_tot = sqrt(n)/2 for n fair coins (worked variant). Morin (2016) eq (3.51)."
