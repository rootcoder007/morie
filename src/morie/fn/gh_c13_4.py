# morie.fn -- function file (rootcoder007/morie)
"""Discrete-time beta process.

Implements sec. 13.3.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bp_discrete"]


def ghosal_bp_discrete(hazards0=(0.1, 0.2, 0.3), c=4.0, seed=42):
    """H(t) = sum_{s<=t} dH(s), dH(s_k) ~ Be(c_k h_k, c_k(1 - h_k))
    independent (sec. 13.3.1): E dH(s_k) = h_k exactly.
    Keys: estimate."""
    h0 = _bnp._flat(hazards0)
    rng = np.random.default_rng(seed)
    n_sim = 2000
    means = [0.0] * len(h0)
    for _ in range(n_sim):
        for k, h in enumerate(h0):
            means[k] += float(rng.beta(c * h, c * (1.0 - h))) / n_sim
    gap = max(abs(m - h) for m, h in zip(means, h0))
    res = RichResult(payload={"estimate": sum(means),
                              "mean_by_time": means,
                              "prior_mean_gap": gap,
                              "method": "discrete beta process (GvdV 2017 sec. 13.3.1)"})
    return with_describe_pointer(res, "gh_c13_4")


def cheatsheet():
    return "gh_c13_4: Discrete-time beta process"
