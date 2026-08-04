# morie.fn -- function file (rootcoder007/morie)
"""DP prior mean.

Implements Proposition 4.2, eq. (4.2) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_mean"]


def ghosal_dp_mean(G0_A):
    """E P(A) = alpha-bar(A) = G0(A) (eq. 4.2). Keys: estimate."""
    v = float(_bnp._flat(G0_A)[0])
    res = RichResult(payload={"estimate": v,
                              "method": "DP mean (GvdV 2017 eq. 4.2)"})
    return with_describe_pointer(res, "gh_c4_2")


def cheatsheet():
    return "gh_c4_2: DP prior mean"


# compact alias per ledger/NAMING.md
ghosaldpmean = ghosal_dp_mean
