# morie.fn -- function file (rootcoder007/morie)
"""Tail-free maximum bound.

Implements eq. (3.14), p.41 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tailfree_max_bound"]


def ghosal_ch3_tailfree_max_bound(EV2_by_level, m):
    """E[max_{e in E^m} P(A_e)]^2 <= sum_{e in E^m} prod_j E(V^2)
    <= 2^m (r/2)^m with r = max 2 E(V^2) (eq. 3.14). Under symmetric
    splits every branch has the same product, so the middle term is
    2^m prod_j E(V_j^2). Keys: value."""
    ev2 = _bnp._flat(EV2_by_level)[:int(m)]
    if len(ev2) < int(m):
        raise ValueError("need E(V^2) for each of the m levels")
    prod = 1.0
    for v in ev2:
        prod *= v
    middle = (2.0 ** int(m)) * prod
    r = max(2.0 * v for v in ev2)
    upper = (2.0 ** int(m)) * (r / 2.0) ** int(m)
    res = RichResult(payload={"estimate": middle, "value": middle,
                              "upper_bound": upper,
                              "bound_holds": middle <= upper + 1e-15,
                              "method": "tail-free max bound (GvdV 2017 eq. 3.14)"})
    return with_describe_pointer(res, "ghs021")


def cheatsheet():
    return "ghs021: Tail-free maximum bound"
