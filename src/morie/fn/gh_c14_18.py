# morie.fn -- function file (rootcoder007/morie)
"""Kernel stick-breaking process.

Implements sec. 14.9.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ksbp_def"]


def ghosal_ksbp_def(x=(0.2, 0.8), n_terms=30, seed=42):
    """p_k(x) = V_k(x) prod_{j<k}(1 - V_j(x)) with covariate-local
    sticks V_k(x) = g(w_k' x) (sec. 14.9.1): weights sum to (near) 1
    at every x and vary smoothly with x. Keys: estimate."""
    rng = np.random.default_rng(seed)
    ws = [(float(rng.normal(0, 1)), float(rng.normal(0, 1)))
          for _ in range(n_terms)]
    out = []
    for xv in _bnp._flat(x):
        V = [1.0 / (1.0 + math.exp(-(a * xv + b))) for a, b in ws]
        p = _bnp.stick_breaking(V)
        out.append(p)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(out[0], out[1]))
    res = RichResult(payload={"estimate": tv,
                              "mass_x0": sum(out[0]),
                              "mass_x1": sum(out[1]),
                              "weights_vary_with_x": tv > 1e-4,
                              "method": "kernel stick-breaking (GvdV 2017 sec. 14.9.1)"})
    return with_describe_pointer(res, "gh_c14_18")


def cheatsheet():
    return "gh_c14_18: Kernel stick-breaking process"
