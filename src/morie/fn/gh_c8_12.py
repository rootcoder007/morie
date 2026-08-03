# morie.fn -- function file (rootcoder007/morie)
"""Minimax lower bound for the rate.

Implements sec. 8.4 (entropy balance; Ex 8.16 rate) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_crt_lower"]


def ghosal_crt_lower(smoothness, n):
    """No procedure beats eps_n = n^{-s/(2s+1)} over an s-Holder
    ball (sec. 8.4): the rate solves the entropy balance
    eps^{-1/s} = n eps^2, and the posterior rate of Thm 8.9 matches
    it -- Bayes is minimax-optimal here. Keys: estimate."""
    s = float(smoothness)
    eps = float(n) ** (-s / (2.0 * s + 1.0))
    gap = abs(eps ** (-1.0 / s) - float(n) * eps ** 2) \
        / (float(n) * eps ** 2)
    res = RichResult(payload={"estimate": eps,
                              "balance_gap": gap,
                              "exponent": s / (2.0 * s + 1.0),
                              "method": "minimax lower bound (GvdV 2017 sec. 8.4)"})
    return with_describe_pointer(res, "gh_c8_12")


def cheatsheet():
    return "gh_c8_12: Minimax lower bound for the rate"
