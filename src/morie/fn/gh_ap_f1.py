# morie.fn -- function file (rootcoder007/morie)
"""Donsker classes.

Implements Appendix F of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_donsker_class"]


def ghosal_donsker_class(smoothness=1.0):
    """F is Donsker if the bracketing integral J_[](1, F, L2) =
    int_0^1 sqrt(log N_[](eps)) deps < infty (App F): with
    log N ~ eps^{-1/s} this holds iff s > 1/2. Keys: estimate."""
    s = float(smoothness)
    # integral of eps^{-1/(2s)} on (0,1): finite iff 1/(2s) < 1
    exponent = 1.0 / (2.0 * s)
    finite = exponent < 1.0
    J = 1.0 / (1.0 - exponent) if finite else float("inf")
    res = RichResult(payload={"estimate": J,
                              "donsker": finite,
                              "method": "Donsker bracketing integral (GvdV 2017 App F)"})
    return with_describe_pointer(res, "gh_ap_f1")


def cheatsheet():
    return "gh_ap_f1: Donsker classes"
