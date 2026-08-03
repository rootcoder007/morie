# morie.fn -- function file (rootcoder007/morie)
"""Entropy condition for a sieve.

Implements eq. (8.5) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_entropy_cnd"]


def ghosal_entropy_cnd(dim, radius, n, eps_n):
    """log N(eps, ball_d(R), ||.||) <= d log(3R/eps) for a
    d-dimensional ball; condition (8.5) requires it <= n eps_n^2.
    Keys: estimate."""
    d = float(dim)
    log_N = d * math.log(max(3.0 * float(radius) / float(eps_n),
                             1.0))
    ok = log_N <= float(n) * float(eps_n) ** 2
    res = RichResult(payload={"estimate": log_N,
                              "bound": float(n) * float(eps_n) ** 2,
                              "condition_holds": ok,
                              "method": "entropy condition (GvdV 2017 eq. 8.5)"})
    return with_describe_pointer(res, "gh_c8_5")


def cheatsheet():
    return "gh_c8_5: Entropy condition for a sieve"
