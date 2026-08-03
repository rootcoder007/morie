# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet aggregation (marginals).

Implements Appendix G (Proposition G.3) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dir_marginal"]


def ghosal_dir_marginal(alpha, merge_idx=(0, 1)):
    """X_{j1} + ... + X_{jm} ~ Be(sum alpha_jl, A - sum alpha_jl)
    (Prop G.3): aggregation of Dirichlet cells stays Beta/Dirichlet.
    Keys: estimate."""
    a = _bnp._flat(alpha)
    A = sum(a)
    s = sum(a[i] for i in merge_idx)
    mean = s / A
    var = s * (A - s) / (A * A * (A + 1.0))
    res = RichResult(payload={"estimate": mean,
                              "beta_params": [s, A - s],
                              "variance": var,
                              "method": "Dirichlet aggregation (GvdV 2017 Prop G.3)"})
    return with_describe_pointer(res, "gh_ap_g3")


def cheatsheet():
    return "gh_ap_g3: Dirichlet aggregation (marginals)"
