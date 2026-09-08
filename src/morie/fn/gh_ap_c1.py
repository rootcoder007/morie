# morie.fn -- function file (rootcoder007/morie)
"""Covering numbers.

Implements Appendix C of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_covering_num"]


def ghosal_covering_num(radius_set=1.0, eps=0.25, dim=2):
    """N(eps, T, d) = minimal number of eps-balls covering T
    (App C): for a d-dim ball of radius R, (R/eps)^d <= N <=
    (3R/eps)^d. Returns the bracket. Keys: estimate."""
    lo = (radius_set / eps) ** dim
    hi = (3.0 * radius_set / eps) ** dim
    res = RichResult(payload={"estimate": hi,
                              "lower": lo, "upper": hi,
                              "log_upper": dim * math.log(
                                  3.0 * radius_set / eps),
                              "method": "covering number bounds (GvdV 2017 App C)"})
    return with_describe_pointer(res, "gh_ap_c1")


def cheatsheet():
    return "gh_ap_c1: Covering numbers"
