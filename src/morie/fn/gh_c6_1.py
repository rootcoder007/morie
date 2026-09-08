# morie.fn -- function file (rootcoder007/morie)
"""Weak posterior consistency.

Implements Definition 6.1 + Proposition 6.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_weak_consist"]


def ghosal_weak_consist(theta0=0.3, eps=0.1, ns=(20, 80, 320, 1280),
                        seed=42):
    """Pi_n(theta: |theta - theta0| > eps | X^n) -> 0 in probability
    (Def 6.1): Beta-Bernoulli posterior mass outside the eps-ball,
    computed exactly by beta-tail quadrature, decreases along n.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    masses = []
    for n in ns:
        S = sum(1 for _ in range(n)
                if float(rng.uniform(0, 1)) < theta0)
        a, b = 1.0 + S, 1.0 + n - S
        # P(|theta-theta0|>eps) by midpoint quadrature of Beta(a,b)
        grid = 3000
        mass = 0.0
        for i in range(grid):
            t = (i + 0.5) / grid
            if abs(t - theta0) > eps:
                mass += math.exp(
                    math.lgamma(a + b) - math.lgamma(a)
                    - math.lgamma(b) + (a - 1.0) * math.log(t)
                    + (b - 1.0) * math.log(1.0 - t)) / grid
        masses.append(mass)
    res = RichResult(payload={"estimate": masses[-1],
                              "mass_outside_by_n": masses,
                              "decreasing": all(
                                  masses[i + 1] <= masses[i] + 1e-9
                                  for i in range(len(masses) - 1)),
                              "method": "weak consistency (GvdV 2017 Def 6.1)"})
    return with_describe_pointer(res, "gh_c6_1")


def cheatsheet():
    return "gh_c6_1: Weak posterior consistency"
