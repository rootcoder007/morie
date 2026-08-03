# morie.fn -- function file (rootcoder007/morie)
"""Metropolis-Hastings sampler.

Implements Appendix M of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mh_sampler"]


def ghosal_mh_sampler(n_draws=4000, seed=42):
    """Accept theta* with prob min(1, pi(theta*) q(theta | theta*) /
    (pi(theta) q(theta* | theta))) (App M): symmetric random-walk MH
    targeting N(0,1); sample moments match. Keys: estimate."""
    rng = np.random.default_rng(seed)
    x = 0.0
    acc = 0
    xs = []
    for _ in range(n_draws):
        prop = x + float(rng.normal(0, 1))
        if math.log(max(float(rng.uniform(0, 1)), 1e-300)) \
                < -0.5 * (prop * prop - x * x):
            x = prop
            acc += 1
        xs.append(x)
    m = sum(xs) / len(xs)
    v = sum((v_ - m) ** 2 for v_ in xs) / (len(xs) - 1)
    res = RichResult(payload={"estimate": v,
                              "mean": m,
                              "accept_rate": acc / n_draws,
                              "method": "Metropolis-Hastings (GvdV 2017 App M)"})
    return with_describe_pointer(res, "gh_ap_m1")


def cheatsheet():
    return "gh_ap_m1: Metropolis-Hastings sampler"
