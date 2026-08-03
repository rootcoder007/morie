# morie.fn -- function file (rootcoder007/morie)
"""Gibbs-type partition processes.

Implements sec. 14.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gibbs_proc"]


def ghosal_gibbs_proc(block_sizes, V_n_k=1.0, discount=0.5):
    """p(n_1..n_k) = V_{n,k} prod_j (1 - d)_{n_j - 1} for Gibbs-type
    processes with discount d (sec. 14.3): the product part uses
    ascending factorials of (1 - d). Keys: estimate."""
    ns = [int(v) for v in _bnp._flat(block_sizes)]
    lp = math.log(float(V_n_k))
    for nj in ns:
        for l in range(nj - 1):
            lp += math.log(1.0 - discount + l)
    res = RichResult(payload={"estimate": math.exp(lp),
                              "log_prob": lp,
                              "method": "Gibbs-type EPPF (GvdV 2017 sec. 14.3)"})
    return with_describe_pointer(res, "gh_c14_8")


def cheatsheet():
    return "gh_c14_8: Gibbs-type partition processes"
