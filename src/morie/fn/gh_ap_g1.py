# morie.fn -- function file (rootcoder007/morie)
"""Finite Dirichlet distribution.

Implements Appendix G of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_fin_dir_def"]


def ghosal_fin_dir_def(alpha=(2.0, 3.0, 5.0), seed=42):
    """p(x) propto prod x_j^{alpha_j - 1} on the simplex (App G):
    draw by gamma normalization; density evaluated at the mean.
    Keys: estimate."""
    a = _bnp._flat(alpha)
    A = sum(a)
    mean = [ai / A for ai in a]
    logC = math.lgamma(A) - sum(math.lgamma(ai) for ai in a)
    logdens = logC + sum((ai - 1.0) * math.log(mi)
                         for ai, mi in zip(a, mean))
    res = RichResult(payload={"estimate": math.exp(logdens),
                              "mean": mean,
                              "log_norm_const": logC,
                              "method": "finite Dirichlet (GvdV 2017 App G)"})
    return with_describe_pointer(res, "gh_ap_g1")


def cheatsheet():
    return "gh_ap_g1: Finite Dirichlet distribution"
