# morie.fn -- function file (rootcoder007/morie)
"""NTR posterior consistency.

Implements sec. 13.4.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ntr_consist"]


def ghosal_ntr_consist(ns=(100, 800, 6400), seed=42):
    """Pi_n(F: d(F, F0) > eps | data) -> 0 under KL-type support for
    NTR priors (sec. 13.4.1). Demo: the Bayesian product-limit
    estimator (unit-exponential BP-type prior) converges to the true
    exponential survival under light censoring. Keys: estimate."""
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        times = []
        events = []
        for _ in range(n):
            x = -math.log(max(float(rng.uniform(0, 1)), 1e-12))
            cens = 3.0 * float(rng.uniform(0, 1))
            times.append(min(x, cens))
            events.append(1.0 if x <= cens else 0.0)
        order = sorted(range(n), key=lambda i: times[i])
        surv = 1.0
        at_risk = n
        S_hat = None
        for i in order:
            if times[i] > 1.0:
                break
            if events[i] > 0:
                surv *= (at_risk - 1.0 + 2.0 * math.exp(-times[i])) \
                    / (at_risk + 2.0 * math.exp(-times[i]))
            at_risk -= 1
        S_hat = surv
        errs.append(abs(S_hat - math.exp(-1.0)))
    res = RichResult(payload={"estimate": errs[-1],
                              "err_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "NTR consistency (GvdV 2017 sec. 13.4.1)"})
    return with_describe_pointer(res, "gh_c13_10")


def cheatsheet():
    return "gh_c13_10: NTR posterior consistency"
