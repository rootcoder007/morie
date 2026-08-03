# morie.fn -- function file (rootcoder007/morie)
"""Consistency for Markov chains.

Implements Theorem 6.42 (transition-KL under the invariant law) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_markov_con"]


def ghosal_markov_con(a0=0.3, b0=0.6, n=3000, seed=42):
    """Markov consistency uses K(p_theta0; p_theta) = int log ratio
    of TRANSITION densities integrated over the invariant measure
    (Thm 6.42). Demo: 2-state chain with flip probabilities (a, b);
    Beta posteriors from transition counts concentrate at (a0, b0).
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    x = 0
    n01 = n00 = n10 = n11 = 0
    for _ in range(n):
        if x == 0:
            if float(rng.uniform(0, 1)) < a0:
                n01 += 1
                x = 1
            else:
                n00 += 1
        else:
            if float(rng.uniform(0, 1)) < b0:
                n10 += 1
                x = 0
            else:
                n11 += 1
    a_hat = (1.0 + n01) / (2.0 + n00 + n01)
    b_hat = (1.0 + n10) / (2.0 + n10 + n11)
    # stationary KL between truth and posterior-mean chain
    pi0 = b0 / (a0 + b0)
    def bkl(p, q):
        return p * math.log(p / q) + (1 - p) * math.log(
            (1 - p) / (1 - q))
    kl = pi0 * bkl(a0, a_hat) + (1 - pi0) * bkl(b0, b_hat)
    res = RichResult(payload={"estimate": kl,
                              "a_hat": a_hat, "b_hat": b_hat,
                              "method": "Markov transition-KL consistency (GvdV 2017 Thm 6.42)"})
    return with_describe_pointer(res, "gh_c6_11")


def cheatsheet():
    return "gh_c6_11: Consistency for Markov chains"
