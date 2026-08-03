# morie.fn -- function file (rootcoder007/morie)
"""Priors on finite approximating nets.

Implements Theorem 8.15 + Example 8.16 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_fin_apx_pri"]


def ghosal_fin_apx_pri(smoothness, n):
    """Net priors: with bracketing entropy log N <= eps^{-1/alpha}
    (Holder ball, Ex 8.16), the rate solves eps^{-1/alpha} = n eps^2:
    eps_n = n^{-alpha/(2 alpha + 1)}, attained by the discrete-net
    mixture prior of Thm 8.15. Keys: estimate."""
    a = float(smoothness)
    eps_n = float(n) ** (-a / (2.0 * a + 1.0))
    balance_gap = abs(eps_n ** (-1.0 / a)
                      - float(n) * eps_n ** 2) \
        / (float(n) * eps_n ** 2)
    res = RichResult(payload={"estimate": eps_n,
                              "net_size_log": eps_n ** (-1.0 / a),
                              "balance_gap": balance_gap,
                              "method": "net-prior rate (GvdV 2017 Thm 8.15, Ex 8.16)"})
    return with_describe_pointer(res, "gh_c8_7")


def cheatsheet():
    return "gh_c8_7: Priors on finite approximating nets"
