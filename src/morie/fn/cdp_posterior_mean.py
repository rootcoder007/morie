# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet posterior mean.

Implements eq. (3.7), p.32 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["cdp_posterior_mean"]


def cdp_posterior_mean(alpha, counts, j, alpha_total):
    """E(p_j | X) = (alpha_j + N_j) / (sum alpha + n) (eq. 3.7).
    Keys: value."""
    v = _bnp.cdp_posterior_mean(alpha, counts, int(j), alpha_total)
    res = RichResult(payload={"estimate": v, "value": v,
                              "method": "Dirichlet posterior mean (GvdV 2017 eq. 3.7)"})
    return with_describe_pointer(res, "ghs014")


def cheatsheet():
    return "ghs014: Dirichlet posterior mean"
