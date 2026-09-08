# morie.fn -- function file (rootcoder007/morie)
"""Local DP regression rate.

Implements sec. 14.9.2 context + rate theory Thm 8.9 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_local_dp_rate"]


def ghosal_local_dp_rate(x=None, s=1.0, ns=(100, 10000, 1000000),
                         log_power=0.5):
    """Covariate-local DP(alpha(x), G0) regression contracts at
    n^{-s/(2s+1)} (log n)^t (local DP of sec. 14.9.2 with the rate
    machinery of Thm 8.9). Computes the rate sequence.
    Keys: estimate."""
    s = float(s)
    rates = [float(n) ** (-s / (2.0 * s + 1.0))
             * math.log(n) ** log_power for n in ns]
    res = RichResult(payload={"estimate": rates[-1],
                              "rate_by_n": rates,
                              "decreasing": all(
                                  rates[i + 1] < rates[i]
                                  for i in range(len(rates) - 1)),
                              "exponent": s / (2.0 * s + 1.0),
                              "method": "local DP rate (GvdV 2017 sec. 14.9.2, Thm 8.9)"})
    return with_describe_pointer(res, "gh_loc_dp_crt")


def cheatsheet():
    return "gh_loc_dp_crt: Local DP regression rate"
