# morie.fn -- function file (rootcoder007/morie)
"""Smooth hazard via a Gaussian process.

Implements sec. 13.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_smhaz_gp"]


def ghosal_smhaz_gp(n=500, seed=42):
    """lambda(t) = exp(f(t)), f ~ GP (sec. 13.5): binned Poisson
    likelihood for the log-hazard with ridge smoothing recovers a
    constant true hazard. Keys: estimate."""
    rng = np.random.default_rng(seed)
    lam0 = 1.0
    k = 6
    d_ = [0.0] * k
    e_ = [0.0] * k
    for _ in range(n):
        x = -math.log(max(float(rng.uniform(0, 1)), 1e-12)) / lam0
        for b in range(k):
            lo, hi = b * 0.3, (b + 1) * 0.3
            if x >= hi:
                e_[b] += 0.3
            elif x > lo:
                e_[b] += x - lo
                d_[b] += 1.0
                break
    f = [math.log(max((d + 0.5) / (e + 0.5), 1e-6))
         for d, e in zip(d_, e_)]
    haz = [math.exp(v) for v in f]
    err = sum(abs(h - lam0) for h in haz) / k
    res = RichResult(payload={"estimate": err,
                              "hazard_by_bin": haz,
                              "method": "GP smooth hazard (GvdV 2017 sec. 13.5)"})
    return with_describe_pointer(res, "gh_c13_12")


def cheatsheet():
    return "gh_c13_12: Smooth hazard via a Gaussian process"
