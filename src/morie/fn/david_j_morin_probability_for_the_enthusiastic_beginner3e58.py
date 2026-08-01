"""Summary triple: sigma_single, sigma_tot, sigma_avg for n biased trials.

Implements eq (3.58) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58(n, p):
    """Summary triple: sigma_single, sigma_tot, sigma_avg for n biased trials.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (3.58).
    """
    sd_single = _morin.sd_bernoulli(p)
    sd_tot = _morin.sd_binomial(n, p)
    sd_avg = sd_tot / int(n)
    payload = {"sd_single": sd_single, "sd_tot": sd_tot, "sd_avg": sd_avg}
    lines = [("sigma_single", sd_single), ("sigma_tot", sd_tot),
             ("sigma_avg", sd_avg)]
    return RichResult(
        title="Summary triple: sigma_single, sigma_tot, sigma_avg for n biased trials.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e58: Summary triple: sigma_single, sigma_tot, sigma_avg for n biased trials. Morin (2016) eq (3.58)."
