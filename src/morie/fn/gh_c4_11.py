# morie.fn -- function file (rootcoder007/morie)
"""Sethuraman stick-breaking representation.

Implements Theorem 4.12 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_stickbr"]


def ghosal_dp_stickbr(n_terms, alpha, seed=42):
    """W_j = V_j prod_{l<j}(1-V_l) with V_j iid Be(1, M), theta_j iid
    G0-bar: sum W_j delta_{theta_j} ~ DP(M G0-bar) (Theorem 4.12).
    Keys: estimate."""
    M = float(alpha)
    rng = np.random.default_rng(seed)
    V = [float(rng.beta(1.0, M)) for _ in range(int(n_terms))]
    W = _bnp.stick_breaking(V)
    th = [float(v) for v in rng.uniform(0, 1, int(n_terms))._flat()]
    mean = sum(wi * t for wi, t in zip(W, th))
    res = RichResult(payload={"estimate": mean, "weights": W[:20],
                              "total_mass": sum(W),
                              "method": "Sethuraman representation (GvdV 2017 Thm 4.12)"})
    return with_describe_pointer(res, "gh_c4_11")


def cheatsheet():
    return "gh_c4_11: Sethuraman stick-breaking representation"
