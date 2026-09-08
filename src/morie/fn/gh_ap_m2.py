# morie.fn -- function file (rootcoder007/morie)
"""Gibbs sampler.

Implements Appendix M of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gibbs_sampler"]


def ghosal_gibbs_sampler(rho=0.6, n_draws=4000, seed=42):
    """theta_j ~ pi(theta_j | theta_{-j}, X) cyclically (App M):
    bivariate normal with correlation rho -- conditionals
    N(rho * other, 1 - rho^2); sample correlation matches rho.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    x = y = 0.0
    xs = []
    ys = []
    s = math.sqrt(1.0 - rho * rho)
    for _ in range(n_draws):
        x = rho * y + s * float(rng.normal(0, 1))
        y = rho * x + s * float(rng.normal(0, 1))
        xs.append(x)
        ys.append(y)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs)
                    * sum((b - my) ** 2 for b in ys))
    res = RichResult(payload={"estimate": num / den,
                              "target_rho": rho,
                              "gap": abs(num / den - rho),
                              "method": "Gibbs sampler (GvdV 2017 App M)"})
    return with_describe_pointer(res, "gh_ap_m2")


def cheatsheet():
    return "gh_ap_m2: Gibbs sampler"
