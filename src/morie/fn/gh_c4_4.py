# morie.fn -- function file (rootcoder007/morie)
"""DP prior covariance.

Implements Proposition 4.2, eq. (4.4) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_cov"]


def ghosal_dp_cov(G0_AB, G0_A, G0_B, alpha):
    """cov(P(A), P(B)) = (G0(A cap B) - G0(A) G0(B)) / (1 + |alpha|)
    (eq. 4.4). Keys: estimate."""
    gab = float(_bnp._flat(G0_AB)[0])
    ga = float(_bnp._flat(G0_A)[0])
    gb = float(_bnp._flat(G0_B)[0])
    M = float(alpha)
    v = (gab - ga * gb) / (1.0 + M)
    res = RichResult(payload={"estimate": v,
                              "method": "DP covariance (GvdV 2017 eq. 4.4)"})
    return with_describe_pointer(res, "gh_c4_4")


def cheatsheet():
    return "gh_c4_4: DP prior covariance"
