# morie.fn -- function file (rootcoder007/morie)
"""Stick-breaking weights.

Implements eq. (3.2), p.30 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_stick_breaking_weights"]


def ghosal_ch3_stick_breaking_weights(V):
    """p_j = (prod_{l<j}(1-V_l)) V_j (eq. 3.2). Keys: distribution."""
    p = _bnp.stick_breaking(V)
    res = RichResult(payload={"estimate": p[0], "distribution": p,
                              "remaining_mass": 1.0 - sum(p),
                              "method": "stick breaking (GvdV 2017 eq. 3.2)"})
    return with_describe_pointer(res, "ghs009")


def cheatsheet():
    return "ghs009: Stick-breaking weights"
