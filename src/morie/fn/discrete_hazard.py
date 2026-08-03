# morie.fn -- function file (rootcoder007/morie)
"""Discrete hazard rate of weights.

Implements eq. (3.3), p.31 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["discrete_hazard"]


def discrete_hazard(p):
    """V_j = p_j / (1 - sum_{l<j} p_l) = P(X=j | X>=j) (eq. 3.3).
    Keys: value."""
    V = _bnp.discrete_hazard(p)
    res = RichResult(payload={"estimate": V[0], "value": V,
                              "method": "discrete hazard (GvdV 2017 eq. 3.3)"})
    return with_describe_pointer(res, "ghs010")


def cheatsheet():
    return "ghs010: Discrete hazard rate of weights"
