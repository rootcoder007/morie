"""Worked sigma of the average: sigma_tot / n = 0.0037.

Implements eq (3.57) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_57(n=10000, p=1.0/6.0):
    """Worked sigma of the average: sigma_tot / n = 0.0037.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.57).
    """
    sd_tot = _morin.sd_binomial(n, p)
    value = sd_tot / int(n)
    check = _morin.sd_of_mean(_morin.sd_bernoulli(p), n)
    if abs(value - check) > 1e-12:
        raise AssertionError("sigma_tot/n != sigma_single/sqrt(n)")
    payload = {"sd_avg": value, "sd_tot": sd_tot}
    lines = [("sigma_avg", value)]
    return RichResult(
        title="Worked sigma of the average: sigma_tot / n = 0.0037.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e57: Worked sigma of the average: sigma_tot / n = 0.0037. Morin (2016) eq (3.57)."
