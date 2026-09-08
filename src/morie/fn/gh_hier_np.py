# morie.fn -- function file (rootcoder007/morie)
"""Hierarchical DP model with random precision.

Implements sec. 4.5 (eq. 4.30 Gibbs for M) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_hierarchical_np"]


def ghosal_hierarchical_np(x=None, a=1.0, b=1.0, n=60, n_sweeps=200,
                           seed=42):
    """theta_i | G ~ G, G ~ DP(alpha, G0), alpha ~ Ga(a, b): alpha is
    updated through the auxiliary-variable Gibbs step (eq. 4.30):
    alpha | eta, K ~ Ga(a + K, b - log eta), eta | alpha ~
    Be(alpha, n) (mixture simplified to the dominant branch).
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    alpha = 1.0
    # simulate a CRP partition once at a fixed true alpha0 = 2
    alpha0 = 2.0
    K = 0
    counts = []
    for i in range(int(n)):
        u = float(rng.uniform(0, 1)) * (alpha0 + i)
        if u < alpha0:
            counts.append(1)
            K += 1
        else:
            acc = alpha0
            for t in range(len(counts)):
                acc += counts[t]
                if u < acc:
                    counts[t] += 1
                    break
    draws = []
    for _ in range(int(n_sweeps)):
        eta = float(rng.beta(max(alpha, 1e-6), float(n)))
        alpha = float(rng.gamma(a + K, 1.0 / (b - math.log(max(
            eta, 1e-12)))))
        draws.append(alpha)
    post_mean = sum(draws[n_sweeps // 4:]) \
        / len(draws[n_sweeps // 4:])
    res = RichResult(payload={"estimate": post_mean,
                              "K_n": K,
                              "posterior_positive": post_mean > 0,
                              "method": "hierarchical DP, alpha Gibbs (GvdV 2017 eq. 4.30)"})
    return with_describe_pointer(res, "gh_hier_np")


def cheatsheet():
    return "gh_hier_np: Hierarchical DP model with random precision"
