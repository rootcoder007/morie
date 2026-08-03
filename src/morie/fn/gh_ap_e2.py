# morie.fn -- function file (rootcoder007/morie)
"""Spline spaces.

Implements Appendix E of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_spline_space"]


def ghosal_spline_space(K=10, order=4, smoothness=2.0):
    """dim(S_{K,r}) = K + r (knots + order) and approximation error
    ||f - s*|| ~ K^{-s} for s-smooth f (App E). Keys: estimate."""
    dim = int(K) + int(order)
    apx = float(K) ** (-smoothness)
    res = RichResult(payload={"estimate": float(dim),
                              "approx_error_order": apx,
                              "method": "spline space (GvdV 2017 App E)"})
    return with_describe_pointer(res, "gh_ap_e2")


def cheatsheet():
    return "gh_ap_e2: Spline spaces"
