# morie.fn -- function file (rootcoder007/morie)
"""Schwartz consistency for i.i.d. data.

Implements Theorem 6.16 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_iid_posterior_consistency"]


def ghosal_iid_posterior_consistency(theta0=0.5, eps=0.2, n=600,
                                     seed=42):
    """Schwartz (Thm 6.16): KL property + exponentially consistent
    tests give strong consistency. For Bernoulli the posterior odds
    of {|theta - theta0| > eps} decay exponentially; the demo reports
    the decay exponent n^{-1} log Pi_n(U^c | X^n). Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = sum(1 for _ in range(n)
            if float(rng.uniform(0, 1)) < theta0)
    a, b = 1.0 + S, 1.0 + n - S
    grid = 4000
    inside = outside = 0.0
    for i in range(grid):
        t = (i + 0.5) / grid
        d = math.exp(math.lgamma(a + b) - math.lgamma(a)
                     - math.lgamma(b) + (a - 1) * math.log(t)
                     + (b - 1) * math.log(1 - t))
        if abs(t - theta0) > eps:
            outside += d
        else:
            inside += d
    mass = outside / (inside + outside)
    exponent = math.log(max(mass, 1e-300)) / n
    res = RichResult(payload={"estimate": mass,
                              "decay_exponent": exponent,
                              "exponential": exponent < 0,
                              "method": "Schwartz consistency (GvdV 2017 Thm 6.16)"})
    return with_describe_pointer(res, "gh_iid_consist")


def cheatsheet():
    return "gh_iid_consist: Schwartz consistency for i.i.d. data"
