# morie.fn -- function file (rootcoder007/morie)
"""DP posterior with censored data.

Implements sec. 13.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_surv_dp_post"]


def ghosal_surv_dp_post(times, events, t_query, alpha=2.0):
    """F | data ~ DP-type posterior accounting for right censoring
    (sec. 13.2): survival via the Bayesian product-limit -- each risk
    set contributes a Beta-posterior factor
    (alpha S0(t_j) + R_j - d_j) / (alpha S0(t_{j-1}) + R_j) with S0
    the prior-mean survival. Keys: estimate."""
    ts = _bnp._flat(times)
    ev = _bnp._flat(events)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    surv = 1.0
    at_risk = len(ts)
    for i in order:
        if ts[i] > t_query:
            break
        S0 = math.exp(-ts[i])              # unit-exponential center
        if ev[i] > 0:
            surv *= (alpha * S0 + at_risk - 1.0) \
                / (alpha * S0 + at_risk)
        at_risk -= 1
    res = RichResult(payload={"estimate": surv,
                              "survival_at_t": surv,
                              "method": "censored DP posterior (GvdV 2017 sec. 13.2)"})
    return with_describe_pointer(res, "gh_c13_1")


def cheatsheet():
    return "gh_c13_1: DP posterior with censored data"
