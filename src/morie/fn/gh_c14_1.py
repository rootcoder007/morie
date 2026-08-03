# morie.fn -- function file (rootcoder007/morie)
"""Exchangeable partition probability function.

Implements sec. 14.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_eppf_def"]


def ghosal_eppf_def(block_sizes, alpha=1.0):
    """p(n_1..n_k) is a symmetric function of the block sizes
    (sec. 14.1): evaluated here for the DP-EPPF (Ewens form, Prop
    4.11): M^k Gamma(M) prod Gamma(n_j) / Gamma(M + n).
    Symmetry under permutation is exact. Keys: estimate."""
    ns = [int(v) for v in _bnp._flat(block_sizes)]
    n = sum(ns)
    k = len(ns)
    lp = k * math.log(alpha) + math.lgamma(alpha) \
        + sum(math.lgamma(v) for v in ns) - math.lgamma(alpha + n)
    lp_perm = k * math.log(alpha) + math.lgamma(alpha) \
        + sum(math.lgamma(v) for v in reversed(ns)) \
        - math.lgamma(alpha + n)
    res = RichResult(payload={"estimate": math.exp(lp),
                              "log_eppf": lp,
                              "symmetric": abs(lp - lp_perm) < 1e-12,
                              "method": "EPPF (GvdV 2017 sec. 14.1, eq. 4.20)"})
    return with_describe_pointer(res, "gh_c14_1")


def cheatsheet():
    return "gh_c14_1: Exchangeable partition probability function"
