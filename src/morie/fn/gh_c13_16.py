# morie.fn -- function file (rootcoder007/morie)
"""Bayesian bootstrap for censored data.

Implements sec. 13.7.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bb_censored"]


def ghosal_bb_censored(times, events, t_query):
    """The alpha -> 0 limit of the censored-data DP posterior is
    Lo's censored Bayesian bootstrap: the product-limit with
    Beta-type weights, reducing to Kaplan-Meier factors
    (R_j - d_j)/R_j at the events (sec. 13.7.1). Keys: estimate."""
    ts = _bnp._flat(times)
    ev = _bnp._flat(events)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    surv = 1.0
    at_risk = len(ts)
    for i in order:
        if ts[i] > t_query:
            break
        if ev[i] > 0:
            surv *= (at_risk - 1.0) / at_risk
        at_risk -= 1
    res = RichResult(payload={"estimate": surv,
                              "km_survival": surv,
                              "method": "censored Bayesian bootstrap = Kaplan-Meier limit (GvdV 2017 sec. 13.7.1)"})
    return with_describe_pointer(res, "gh_c13_16")


def cheatsheet():
    return "gh_c13_16: Bayesian bootstrap for censored data"
