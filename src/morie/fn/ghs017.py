# morie.fn -- function file (rootcoder007/morie)
"""Discrete random measure.

Implements sec. 3.3, p.30 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_discrete_random_measure"]


def ghosal_ch3_discrete_random_measure(W, theta):
    """P = sum_i W_i delta_{theta_i}: normalized weights on atoms.
    Keys: distribution."""
    w = _bnp.normalize_weights(W)
    th = _bnp._flat(theta)
    mean = sum(wi * t for wi, t in zip(w, th))
    res = RichResult(payload={"estimate": mean,
                              "distribution": list(zip(th, w)),
                              "total_mass": sum(w),
                              "method": "discrete random measure (GvdV 2017 sec. 3.3)"})
    return with_describe_pointer(res, "ghs017")


def cheatsheet():
    return "ghs017: Discrete random measure"
