# morie.fn -- function file (rootcoder007/morie)
"""Strong-support event product.

Implements Theorem 3.19, eq. (3.20), p.47 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tailfree_strong_support_event"]


def ghosal_ch3_tailfree_strong_support_event(prob_ratio_event,
                                             prob_pm_event):
    """Pi(int |p/p_m - 1| dmu < eps/(2||p0||_inf + eps)) *
    Pi(||p_m - p0||_inf < eps/2) (eq. 3.20): the two independent
    events whose product lower-bounds the prior mass of a total
    variation neighbourhood -- independence is exactly tail-freeness.
    Keys: value."""
    p1 = float(_bnp._flat(prob_ratio_event)[0])
    p2 = float(_bnp._flat(prob_pm_event)[0])
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must lie in [0,1]")
    res = RichResult(payload={"estimate": p1 * p2, "value": p1 * p2,
                              "positive": p1 * p2 > 0,
                              "method": "strong support lower bound (GvdV 2017 eq. 3.20)"})
    return with_describe_pointer(res, "ghs027")


def cheatsheet():
    return "ghs027: Strong-support event product"
