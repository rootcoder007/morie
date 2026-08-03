# morie.fn -- function file (rootcoder007/morie)
"""Cox model with a beta-process baseline.

Implements sec. 13.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_cox_model"]


def ghosal_cox_model(beta=0.7, z=(0.0, 1.0), t=1.0, c=2.0):
    """lambda(t | z) = lambda_0(t) e^{beta z}, lambda_0 ~ BP, beta ~
    Normal (sec. 13.6): prior-mean cumulative hazard for each
    covariate value is H0(t) e^{beta z} -- proportional hazards by
    construction. Keys: estimate."""
    zs = _bnp._flat(z)
    H = [t * math.exp(beta * zi) for zi in zs]
    ratio = H[1] / H[0]
    res = RichResult(payload={"estimate": ratio,
                              "cum_hazards": H,
                              "proportional": abs(ratio
                                                  - math.exp(beta))
                              < 1e-12,
                              "method": "Cox with BP baseline (GvdV 2017 sec. 13.6)"})
    return with_describe_pointer(res, "gh_c13_13")


def cheatsheet():
    return "gh_c13_13: Cox model with a beta-process baseline"
