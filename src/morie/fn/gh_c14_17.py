# morie.fn -- function file (rootcoder007/morie)
"""Hierarchy of discrete random measures.

Implements sec. 14.8 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_disc_rp_rel"]


def ghosal_disc_rp_rel(d=0.0, theta=1.0):
    """DP subset PY subset PK subset NCRM (sec. 14.8): PY(0, theta)
    IS DP(theta) -- the stick law Beta(1, theta + 0k) = Beta(1,
    theta) recovers Sethuraman. Checks the reduction at d = 0.
    Keys: estimate."""
    is_dp = abs(d) < 1e-15
    stick_a = 1.0 - d
    stick_b_k1 = theta + d
    res = RichResult(payload={"estimate": 1.0 if is_dp else 0.0,
                              "py_reduces_to_dp": is_dp
                              and abs(stick_a - 1.0) < 1e-15
                              and abs(stick_b_k1 - theta) < 1e-15,
                              "hierarchy": ["DP", "PY", "PK", "NCRM"],
                              "method": "discrete-measure hierarchy (GvdV 2017 sec. 14.8)"})
    return with_describe_pointer(res, "gh_c14_17")


def cheatsheet():
    return "gh_c14_17: Hierarchy of discrete random measures"
