# morie.fn -- function file (rootcoder007/morie)
"""Tree splitting variables.

Implements eq. (3.11) context, sec. 3.6, p.39 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tree_splitting_variables"]


def ghosal_ch3_tree_splitting_variables(A_epsilon, epsilon=None):
    """V_{e0} = P(A_{e0} | A_e), V_{e1} = P(A_{e1} | A_e): conditional
    child masses given (mass_parent, mass_child0, mass_child1).
    Keys: value."""
    m = _bnp._flat(A_epsilon)
    if len(m) != 3:
        raise ValueError("need (parent, child0, child1) masses")
    parent, c0, c1 = m
    if parent <= 0:
        raise ValueError("parent mass must be positive")
    V0, V1 = c0 / parent, c1 / parent
    res = RichResult(payload={"estimate": V0,
                              "value": [V0, V1],
                              "complement_gap": abs(V0 + V1 - 1.0)
                              if abs(c0 + c1 - parent) < 1e-12 else None,
                              "method": "splitting variables (GvdV 2017 sec. 3.6)"})
    return with_describe_pointer(res, "ghs018")


def cheatsheet():
    return "ghs018: Tree splitting variables"
