# morie.fn -- function file (rootcoder007/morie)
"""Consistency for independent non-i.i.d. data.

Implements Theorem 6.41 (eq. 6.8 average-KL condition) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_non_iid_con"]


def ghosal_non_iid_con(mu0=1.0, ns=(30, 120, 480), seed=42):
    """Triangular arrays: consistency needs the AVERAGE divergence
    sup_B n^{-1} sum_i K(p_{0,i}; p_i) <= eps plus the V_{2,0}
    control (Thm 6.41). Demo: X_i ~ N(mu0, sigma_i^2) with known
    heteroscedastic sigma_i; the weighted-mean posterior for mu
    concentrates. Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        prec = 0.0
        wsum = 0.0
        for i in range(n):
            s = 1.0 + (i % 3)             # sigma in {1, 2, 3}
            x = mu0 + s * float(rng.normal(0, 1))
            prec += 1.0 / s ** 2
            wsum += x / s ** 2
        post_mean = wsum / (prec + 1.0)   # N(0,1) prior on mu
        errs.append(abs(post_mean - mu0))
    # average per-observation KL between N(mu0,s) and N(mu,s):
    # (mu-mu0)^2/(2s^2) -- bounded average, condition (6.8) form
    avg_kl_at_01 = sum(0.01 / (2.0 * (1.0 + (i % 3)) ** 2)
                       for i in range(30)) / 30.0
    res = RichResult(payload={"estimate": errs[-1],
                              "error_by_n": errs,
                              "avg_kl_at_delta_0.1": avg_kl_at_01,
                              "method": "non-iid average-KL consistency (GvdV 2017 Thm 6.41)"})
    return with_describe_pointer(res, "gh_c6_10")


def cheatsheet():
    return "gh_c6_10: Consistency for independent non-i.i.d. data"
