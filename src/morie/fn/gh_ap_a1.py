# morie.fn -- function file (rootcoder007/morie)
"""Weak convergence of measures.

Implements Appendix A of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_weak_conv_def"]


def ghosal_weak_conv_def(p_seq_param=(0.4, 0.45, 0.49, 0.499),
                         p_limit=0.5):
    """P_n -> P weakly iff E_{P_n} f -> E_P f for all bounded
    continuous f (App A). Bernoulli(p_n) -> Bernoulli(p): check on
    the test functions f(x) = x and f(x) = cos(x). Keys: estimate."""
    ps = _bnp._flat(p_seq_param)
    gaps = []
    for f in (lambda x: x, math.cos):
        lim = (1.0 - p_limit) * f(0.0) + p_limit * f(1.0)
        vals = [(1.0 - p) * f(0.0) + p * f(1.0) for p in ps]
        gaps.append(abs(vals[-1] - lim))
    res = RichResult(payload={"estimate": max(gaps),
                              "converging": max(gaps) < 0.01,
                              "method": "weak convergence (GvdV 2017 App A)"})
    return with_describe_pointer(res, "gh_ap_a1")


def cheatsheet():
    return "gh_ap_a1: Weak convergence of measures"
