# morie.fn -- function file (rootcoder007/morie)
"""Kullback-Leibler divergence.

Implements Definition 6.15 context (K(p0; p) = int p0 log(p0/p)) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kl_diverge"]


def ghosal_kl_diverge(p0, p):
    """K(p0; p) = int log(p0/p) dP0 for discrete densities.
    Keys: estimate."""
    p0 = _bnp.normalize_weights(p0)
    p = _bnp.normalize_weights(p)
    kl = 0.0
    for q, pi in zip(p0, p):
        if q > 0:
            if pi <= 0:
                kl = float("inf")
                break
            kl += q * math.log(q / pi)
    res = RichResult(payload={"estimate": kl,
                              "nonnegative": kl >= -1e-15,
                              "method": "KL divergence (GvdV 2017 sec. 6.4)"})
    return with_describe_pointer(res, "gh_c6_7")


def cheatsheet():
    return "gh_c6_7: Kullback-Leibler divergence"
