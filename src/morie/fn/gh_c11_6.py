# morie.fn -- function file (rootcoder007/morie)
"""Brownian-motion prior.

Implements Example 11.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bm_prior"]


def ghosal_bm_prior(n_grid=200, n_sim=400, seed=42):
    """BM: E W(t) = 0, Cov(W(s), W(t)) = min(s, t), paths
    Holder-alpha for alpha < 1/2 (Ex 11.5). Empirical covariance of
    simulated paths matches min(s,t). Keys: estimate."""
    rng = np.random.default_rng(seed)
    s_idx, t_idx = n_grid // 4, n_grid // 2
    acc_st = acc_ss = 0.0
    for _ in range(n_sim):
        w = 0.0
        ws = wt = 0.0
        for i in range(1, n_grid + 1):
            w += float(rng.normal(0, 1)) / math.sqrt(n_grid)
            if i == s_idx:
                ws = w
            if i == t_idx:
                wt = w
        acc_st += ws * wt / n_sim
        acc_ss += ws * ws / n_sim
    s = s_idx / n_grid
    res = RichResult(payload={"estimate": acc_st,
                              "theory_min_st": s,
                              "cov_gap": abs(acc_st - s),
                              "var_gap": abs(acc_ss - s),
                              "method": "Brownian motion prior (GvdV 2017 Ex 11.5)"})
    return with_describe_pointer(res, "gh_c11_6")


def cheatsheet():
    return "gh_c11_6: Brownian-motion prior"
