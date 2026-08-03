# morie.fn -- function file (rootcoder007/morie)
"""Bracketing numbers.

Implements Appendix C of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bracket_num"]


def ghosal_bracket_num(smoothness=1.0, eps=0.1):
    """N_[](eps, T, d): minimal eps-brackets covering T (App C); for
    an s-Holder unit ball on [0,1], log N_[] ~ eps^{-1/s}.
    Keys: estimate."""
    log_N = eps ** (-1.0 / smoothness)
    res = RichResult(payload={"estimate": log_N,
                              "entropy_exponent": 1.0 / smoothness,
                              "method": "bracketing entropy (GvdV 2017 App C)"})
    return with_describe_pointer(res, "gh_ap_c3")


def cheatsheet():
    return "gh_ap_c3: Bracketing numbers"
