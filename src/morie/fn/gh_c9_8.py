# morie.fn -- function file (rootcoder007/morie)
"""Nonlinear autoregression contraction.

Implements sec. 9.5.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_nlar_crt"]


def ghosal_nlar_crt(ns=(200, 800, 3200), seed=42):
    """X_t = f(X_{t-1}) + e_t with f smooth: GP-type posterior
    contracts at n^{-s/(2s+1)} under ergodicity (sec. 9.5.3). Binned
    conditional-mean posterior (normal-normal per bin) recovers
    f(x) = 0.5 x with falling error. Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        x = 0.0
        k = 8
        s_ = [0.0] * k
        c_ = [0.0] * k
        for _ in range(n):
            nxt = 0.5 * x + 0.5 * float(rng.normal(0, 1))
            b = min(max(int((x + 3.0) / 0.75), 0), k - 1)
            s_[b] += nxt
            c_[b] += 1.0
            x = nxt
        err = 0.0
        for b in range(k):
            centre = -3.0 + (b + 0.5) * 0.75
            post = s_[b] / (c_[b] + 1.0)          # N(0,.) prior
            err += abs(post - 0.5 * centre) / k
        errs.append(err)
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "nonlinear AR contraction (GvdV 2017 sec. 9.5.3)"})
    return with_describe_pointer(res, "gh_c9_8")


def cheatsheet():
    return "gh_c9_8: Nonlinear autoregression contraction"
