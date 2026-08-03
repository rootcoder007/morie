# morie.fn -- function file (rootcoder007/morie)
"""Contraction-rate definition.

Implements Definition 8.1 + Lemma 8.2 + Example 8.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_crt_def"]


def ghosal_crt_def(theta0=0.4, n=400, M_list=(1.0, 3.0, 9.0),
                   seed=42):
    """eps_n is a contraction rate if Pi_n(d(theta, theta0) >=
    M_n eps_n | X^n) -> 0 for every M_n -> infty (Def 8.1). For
    Bernoulli the rate is eps_n = n^{-1/2} (Ex 8.3): posterior mass
    outside M eps_n falls in M by Chebyshev via Lemma 8.2.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    S = sum(1 for _ in range(n)
            if float(rng.uniform(0, 1)) < theta0)
    a, b = 1.0 + S, 1.0 + n - S
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    eps = 1.0 / math.sqrt(n)
    masses = []
    for M in M_list:
        # Chebyshev bound around the posterior mean (Lemma 8.2)
        masses.append(min(1.0, var / (M * eps) ** 2
                          + (1.0 if abs(mean - theta0) > M * eps
                             else 0.0)))
    res = RichResult(payload={"estimate": masses[-1],
                              "mass_bound_by_M": masses,
                              "eps_n": eps,
                              "decreasing_in_M": all(
                                  masses[i + 1] <= masses[i] + 1e-12
                                  for i in range(len(masses) - 1)),
                              "method": "contraction rate via Chebyshev (GvdV 2017 Def 8.1, Lemma 8.2)"})
    return with_describe_pointer(res, "gh_c8_1")


def cheatsheet():
    return "gh_c8_1: Contraction-rate definition"
