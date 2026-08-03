# morie.fn -- function file (rootcoder007/morie)
"""Sup-norm posterior contraction.

Implements sec. 11.3.2 context (Thm 11.22-11.24 family) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_sup_norm_contraction"]


def ghosal_sup_norm_contraction(s=1.0, d=1.0, log_power=0.5,
                                ns=(100, 10000, 1000000)):
    """Pi_n(||f - f0||_infty > M eps_n | data) -> 0 with eps_n =
    n^{-s/(2s+d)} (log n)^t for suitable GP priors (sec. 11.3-11.4).
    Computes the rate sequence. Keys: estimate."""
    s = float(s)
    d = float(d)
    rates = [float(n) ** (-s / (2.0 * s + d))
             * math.log(n) ** log_power for n in ns]
    res = RichResult(payload={"estimate": rates[-1],
                              "rate_by_n": rates,
                              "decreasing": all(
                                  rates[i + 1] < rates[i]
                                  for i in range(len(rates) - 1)),
                              "method": "sup-norm contraction rate (GvdV 2017 sec. 11.3)"})
    return with_describe_pointer(res, "gh_sup_norm_gp")


def cheatsheet():
    return "gh_sup_norm_gp: Sup-norm posterior contraction"
