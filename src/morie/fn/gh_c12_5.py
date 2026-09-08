# morie.fn -- function file (rootcoder007/morie)
"""Efficient influence function.

Implements sec. 12.3.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_eff_infl_fn"]


def ghosal_eff_infl_fn(data, t):
    """psi-tilde(x) = 1{x <= t} - F0(t): mean zero under P0, variance
    F0(t)(1 - F0(t)) = the efficient variance for estimating F(t)
    (sec. 12.3.1). Empirical check on data. Keys: estimate."""
    xs = _bnp._flat(data)
    n = len(xs)
    F_t = sum(1 for v in xs if v <= t) / n
    infl = [(1.0 if v <= t else 0.0) - F_t for v in xs]
    mean0 = sum(infl) / n
    var = sum(v * v for v in infl) / n
    res = RichResult(payload={"estimate": var,
                              "mean_zero_gap": abs(mean0),
                              "matches_bernoulli_var":
                                  abs(var - F_t * (1 - F_t)) < 1e-12,
                              "method": "efficient influence function (GvdV 2017 sec. 12.3.1)"})
    return with_describe_pointer(res, "gh_c12_5")


def cheatsheet():
    return "gh_c12_5: Efficient influence function"
