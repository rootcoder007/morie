# morie.fn -- function file (rootcoder007/morie)
"""Penalized Dirichlet process.

Implements sec. 4.6.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_pen_dp"]


def ghosal_pen_dp(p, alpha, lam, counts=None):
    """Density proportional to prod p_j^{alpha_j - 1} exp(-lambda
    Delta(p)) with roughness Delta(p) = sum (p_{j+1} - p_j)^2
    (sec. 4.6.3): evaluates the log prior density; the posterior has
    the same form with alpha_j -> alpha_j + N_j. Keys: estimate."""
    ps = _bnp._flat(p)
    a = _bnp._flat(alpha)
    if abs(sum(ps) - 1.0) > 1e-9:
        raise ValueError("p must lie on the simplex")
    if counts is not None:
        a = [ai + Ni for ai, Ni in zip(a, _bnp._flat(counts))]
    pen = sum((ps[j + 1] - ps[j]) ** 2 for j in range(len(ps) - 1))
    logdens = sum((ai - 1.0) * math.log(max(pi, 1e-300))
                  for ai, pi in zip(a, ps)) - float(lam) * pen
    res = RichResult(payload={"estimate": logdens,
                              "penalty": pen,
                              "posterior_alpha": a,
                              "method": "penalized Dirichlet log density (GvdV 2017 sec. 4.6.3)"})
    return with_describe_pointer(res, "gh_c4_23")


def cheatsheet():
    return "gh_c4_23: Penalized Dirichlet process"
