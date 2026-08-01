"""Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.

Implements eq (4.95) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_95(n, p):
    """Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.95).
    """
    k = int(round(float(p) * int(n)))
    value = _morin.poisson_pmf(k, float(p) * int(n))
    payload = {"k": k, "PP": value}
    lines = [("PP(pn)", value)]
    return RichResult(
        title="Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e95: Poisson peak value PP(pn) = (pn)^(pn) e^(-pn) / (pn)!. Morin (2016) eq (4.95)."
