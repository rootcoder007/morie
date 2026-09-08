# morie.fn -- function file (rootcoder007/morie)
"""Bernstein polynomial approximation.

Implements Appendix E of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bernstein_poly"]


def ghosal_bernstein_poly(K_list=(5, 20, 80)):
    """||B_K f - f||_infty <= C omega(f, 1/sqrt(K)) (App E): for
    f(x) = |x - 1/2| the Bernstein error decays like K^{-1/2}.
    Uses the certified operator from _bnp (sec. 2.3.4 form on the
    CDF scale, applied to f directly). Keys: estimate."""
    f = lambda x: abs(x - 0.5)
    errs = []
    for K in K_list:
        err = 0.0
        for j in range(21):
            x = j / 20.0
            bk = sum(f(k / K) * math.comb(K, k) * x ** k
                     * (1.0 - x) ** (K - k) for k in range(K + 1))
            err = max(err, abs(bk - f(x)))
        errs.append(err)
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_K": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "Bernstein approximation (GvdV 2017 App E)"})
    return with_describe_pointer(res, "gh_ap_e1")


def cheatsheet():
    return "gh_ap_e1: Bernstein polynomial approximation"
