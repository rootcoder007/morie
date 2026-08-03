# morie.fn -- function file (rootcoder007/morie)
"""Epsilon-Dirichlet finite approximation.

Implements eq. (4.23) + Proposition 4.20 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_fs_approx"]


def ghosal_dp_fs_approx(eps, alpha, seed=42):
    """P_eps keeps stick-breaking atoms until sum W_i > 1 - eps and
    lumps the rest on one extra point (eq. 4.23); d_TV(P_eps, P) <=
    eps a.s. and N_eps - 1 ~ Poi(M log_(eps)) so
    E(N_eps + 1) = 2 + M log_(eps) (Prop 4.20). Keys: estimate."""
    eps = float(eps)
    M = float(alpha)
    rng = np.random.default_rng(seed)
    W = []
    left = 1.0
    while left > eps and len(W) < 100000:
        V = float(rng.beta(1.0, M))
        W.append(left * V)
        left *= (1.0 - V)
    N_eps = len(W)
    expected_terms = 2.0 + M * (-math.log(eps))
    res = RichResult(payload={"estimate": float(N_eps),
                              "tv_bound": eps,
                              "remainder_mass": left,
                              "expected_support_size": expected_terms,
                              "method": "eps-Dirichlet process (GvdV 2017 eq. 4.23, Prop 4.20)"})
    return with_describe_pointer(res, "gh_c4_14")


def cheatsheet():
    return "gh_c4_14: Epsilon-Dirichlet finite approximation"
