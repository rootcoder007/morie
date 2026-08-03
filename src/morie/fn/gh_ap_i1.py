# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov continuity for GPs.

Implements Appendix I of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_sample_cont"]


def ghosal_gp_sample_cont(p=2.0, alpha_exc=1.0):
    """E|f(x) - f(y)|^p <= C ||x - y||^{1 + alpha} implies
    Holder-(alpha/p) continuous paths (App I): for BM,
    E|B_s - B_t|^2 = |s - t| gives alpha/p exponent -> any order
    < 1/2 after iterating p. Keys: estimate."""
    holder = alpha_exc / p
    # BM: with p = 2m, E|inc|^{2m} = C |s-t|^m -> exponent (m-1)/2m
    # -> 1/2 as m grows
    limit = 0.5
    res = RichResult(payload={"estimate": holder,
                              "bm_limit_exponent": limit,
                              "method": "Kolmogorov continuity (GvdV 2017 App I)"})
    return with_describe_pointer(res, "gh_ap_i1")


def cheatsheet():
    return "gh_ap_i1: Kolmogorov continuity for GPs"
