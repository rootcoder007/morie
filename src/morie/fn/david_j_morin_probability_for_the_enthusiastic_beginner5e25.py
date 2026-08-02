"""Gaussian tail at 20 sigma: f(20s)*s = e^(-200)/sqrt(2pi) ~ 1e-87.

Implements eq (5.25) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25(n_sigmas=20.0, sigma=1.0):
    """Gaussian tail at 20 sigma: f(20s)*s = e^(-200)/sqrt(2pi) ~ 1e-87.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (5.25).
    """
    x = float(n_sigmas) * float(sigma)
    value = _morin.normal_pdf(x, 0.0, sigma) * float(sigma)
    payload = {"n_sigmas": float(n_sigmas), "area_fraction": value}
    lines = [("sigma * f(x)", value)]
    return RichResult(
        title="Gaussian tail at 20 sigma: f(20s)*s = e^(-200)/sqrt(2pi) ~ 1e-87.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner5e25: Gaussian tail at 20 sigma: f(20s)*s = e^(-200)/sqrt(2pi) ~ 1e-87. Morin (2016) eq (5.25)."
