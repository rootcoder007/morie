# morie.fn -- function file (rootcoder007/morie)
"""Invariant (symmetrized) Dirichlet process.

Implements Definition 4.32 + eq. (4.34) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_inv_dp"]


def ghosal_inv_dp(x, data, alpha_x, alpha_total):
    """Symmetrized DP posterior mean of F (eq. 4.34):
    E(F(x)|X) = (alpha(x) + (1/2) sum (1{X_i <= x} + 1{X_i >= -x}))
    / (|alpha| + n) -- reflection-group invariant version.
    Keys: estimate."""
    xs = _bnp._flat(data)
    x = float(x)
    n = len(xs)
    count = sum(0.5 * ((1.0 if v <= x else 0.0)
                       + (1.0 if v >= -x else 0.0)) for v in xs)
    est = (float(alpha_x) + count) / (float(alpha_total) + n)
    res = RichResult(payload={"estimate": est, "n": n,
                              "symmetrized_count": count,
                              "method": "invariant DP posterior mean (GvdV 2017 eq. 4.34)"})
    return with_describe_pointer(res, "gh_c4_21")


def cheatsheet():
    return "gh_c4_21: Invariant (symmetrized) Dirichlet process"
