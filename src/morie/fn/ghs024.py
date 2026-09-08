# morie.fn -- function file (rootcoder007/morie)
"""Canonical summability conditions.

Implements Theorem 3.16, eq. (3.17), p.44 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tailfree_canonical_summability"]


def ghosal_ch3_tailfree_canonical_summability(EV_by_level,
                                              varV_by_level):
    """sum_m max_e |E(V) - 1/2| < infty and sum_m max_e var(V) < infty
    (eq. 3.17) -- the two series that guarantee absolute continuity
    for canonical partitions. Keys: value."""
    ev = _bnp._flat(EV_by_level)
    vv = _bnp._flat(varV_by_level)
    s1 = sum(abs(v - 0.5) for v in ev)
    s2 = sum(vv)
    res = RichResult(payload={"estimate": s1 + s2,
                              "value": [s1, s2],
                              "mean_series": s1, "var_series": s2,
                              "summable": math.isfinite(s1)
                              and math.isfinite(s2),
                              "method": "canonical summability (GvdV 2017 eq. 3.17)"})
    return with_describe_pointer(res, "ghs024")


def cheatsheet():
    return "ghs024: Canonical summability conditions"
