# morie.fn -- function file (rootcoder007/morie)
"""Slice sampler.

Implements Appendix M of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_slice_sampler"]


def ghosal_slice_sampler(n_draws=4000, seed=42):
    """u ~ Unif(0, pi(theta)), theta ~ Unif{pi > u} (App M): slice
    sampling the Exp(1) density pi(t) = e^{-t}: the slice is
    (0, -log u'); sample mean matches 1. Keys: estimate."""
    rng = np.random.default_rng(seed)
    t = 1.0
    ts = []
    for _ in range(n_draws):
        u = float(rng.uniform(0, 1)) * math.exp(-t)
        upper = -math.log(max(u, 1e-300))
        t = float(rng.uniform(0, 1)) * upper
        ts.append(t)
    m = sum(ts) / len(ts)
    res = RichResult(payload={"estimate": m,
                              "target_mean": 1.0,
                              "gap": abs(m - 1.0),
                              "method": "slice sampler (GvdV 2017 App M)"})
    return with_describe_pointer(res, "gh_ap_m3")


def cheatsheet():
    return "gh_ap_m3: Slice sampler"
