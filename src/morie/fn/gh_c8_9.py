# morie.fn -- function file (rootcoder007/morie)
"""Markov-chain contraction rate.

Implements sec. 8.3.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_markov_crt"]


def ghosal_markov_crt(a0=0.3, b0=0.5, ns=(200, 800, 3200), seed=42):
    """Markov transition estimation: with Beta priors on the flip
    probabilities the posterior risk of the transition matrix falls
    at the parametric n^{-1} rate (sec. 8.3.3, prior mass + entropy
    on the transition class). Keys: estimate."""
    rng = np.random.default_rng(seed)
    risks = []
    for n in ns:
        x = 0
        c = [[0, 0], [0, 0]]
        for _ in range(n):
            p = a0 if x == 0 else b0
            nxt = 1 - x if float(rng.uniform(0, 1)) < p else x
            c[x][1 if nxt != x else 0] += 1
            x = nxt
        va = ((1 + c[0][1]) * (1 + c[0][0])
              / ((2 + sum(c[0])) ** 2 * (3 + sum(c[0]))))
        vb = ((1 + c[1][1]) * (1 + c[1][0])
              / ((2 + sum(c[1])) ** 2 * (3 + sum(c[1]))))
        risks.append(va + vb)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    res = RichResult(payload={"estimate": rate_hat,
                              "posterior_var_by_n": risks,
                              "method": "Markov contraction (GvdV 2017 sec. 8.3.3)"})
    return with_describe_pointer(res, "gh_c8_9")


def cheatsheet():
    return "gh_c8_9: Markov-chain contraction rate"
