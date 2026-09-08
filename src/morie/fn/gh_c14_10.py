# morie.fn -- function file (rootcoder007/morie)
"""Pitman-Yor EPPF.

Implements sec. 14.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_py_eppf"]


def ghosal_py_eppf(block_sizes, d=0.5, theta=1.0):
    """p(n_1..n_k) = [prod_{j<k}(theta + j d)] / (theta + 1)^{[n-1]}
    * prod_j (1 - d)^{[n_j - 1]} (sec. 14.4). Keys: estimate."""
    ns = [int(v) for v in _bnp._flat(block_sizes)]
    n = sum(ns)
    k = len(ns)
    lp = 0.0
    for j in range(1, k):
        lp += math.log(theta + j * d)
    for i in range(1, n):
        lp -= math.log(theta + i)
    for nj in ns:
        for l in range(nj - 1):
            lp += math.log(1.0 - d + l)
    res = RichResult(payload={"estimate": math.exp(lp),
                              "log_eppf": lp,
                              "method": "PY EPPF (GvdV 2017 sec. 14.4)"})
    return with_describe_pointer(res, "gh_c14_10")


def cheatsheet():
    return "gh_c14_10: Pitman-Yor EPPF"


# compact alias per ledger/NAMING.md
ghosalpyeppf = ghosal_py_eppf
