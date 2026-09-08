# morie.fn -- function file (rootcoder007/morie)
"""Borell-TIS inequality.

Implements Appendix I of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_borell_tis"]


def ghosal_borell_tis(u=2.0, sigma_f=1.0):
    """P(sup f - E sup f > u) <= exp(-u^2/(2 sigma_f^2)) (App I):
    Gaussian concentration of the supremum. Returns the bound and
    monotonicity in u. Keys: estimate."""
    b = math.exp(-u * u / (2.0 * sigma_f ** 2))
    b2 = math.exp(-(u + 1.0) ** 2 / (2.0 * sigma_f ** 2))
    res = RichResult(payload={"estimate": b,
                              "tighter_for_larger_u": b2 < b,
                              "method": "Borell-TIS bound (GvdV 2017 App I)"})
    return with_describe_pointer(res, "gh_ap_i3")


def cheatsheet():
    return "gh_ap_i3: Borell-TIS inequality"
