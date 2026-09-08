# morie.fn -- function file (rootcoder007/morie)
"""Weak convergence of Dirichlet processes.

Implements Theorem 4.16 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_weak_conv"]


def ghosal_dp_weak_conv(G0_A, alpha_seq):
    """DP(alpha_m) converges weakly as |alpha_m| -> 0 (to delta_X,
    X ~ G0-bar), to DP(alpha) for a finite limit, or to
    delta_{G0-bar} as |alpha_m| -> infty (Theorem 4.16). Reported via
    the variance of P(A): G0(1-G0)/(1+M_m), which tends to the
    Bernoulli variance, a positive limit, or zero respectively.
    Keys: estimate."""
    g = float(_bnp._flat(G0_A)[0])
    Ms = _bnp._flat(alpha_seq)
    vars_ = [g * (1.0 - g) / (1.0 + m) for m in Ms]
    if Ms[-1] < 1e-6:
        regime = "degenerate at random point (var -> G0(1-G0))"
    elif Ms[-1] > 1e6:
        regime = "degenerate at center measure (var -> 0)"
    else:
        regime = "DP limit (var positive)"
    res = RichResult(payload={"estimate": vars_[-1],
                              "var_sequence": vars_, "regime": regime,
                              "method": "DP weak convergence regimes (GvdV 2017 Thm 4.16)"})
    return with_describe_pointer(res, "gh_c4_13")


def cheatsheet():
    return "gh_c4_13: Weak convergence of Dirichlet processes"
