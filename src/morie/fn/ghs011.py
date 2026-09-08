# morie.fn -- function file (rootcoder007/morie)
"""Countable Dirichlet marginal.

Implements eq. (3.4), p.31 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_countable_dirichlet_marginal"]


def ghosal_ch3_countable_dirichlet_marginal(alpha, k):
    """(p_1..p_k, 1-sum) ~ Dir(k+1; alpha_1..alpha_k, tail alpha)
    (eq. 3.4). Returns the finite Dirichlet parameter vector.
    Keys: distribution."""
    a = _bnp._flat(alpha)
    k = int(k)
    head = a[:k]
    tail = sum(a[k:])
    params = head + [tail]
    tot = sum(a)
    means = [ai / tot for ai in params]
    res = RichResult(payload={"estimate": means[0],
                              "distribution": params,
                              "mean": means,
                              "method": "countable Dirichlet marginal (GvdV 2017 eq. 3.4)"})
    return with_describe_pointer(res, "ghs011")


def cheatsheet():
    return "ghs011: Countable Dirichlet marginal"
