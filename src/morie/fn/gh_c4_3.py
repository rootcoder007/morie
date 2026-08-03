# morie.fn -- function file (rootcoder007/morie)
"""DP prior variance.

Implements Proposition 4.2, eq. (4.3) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_var"]


def ghosal_dp_var(G0_A, alpha):
    """var P(A) = G0(A)(1-G0(A)) / (1 + |alpha|) (eq. 4.3).
    Keys: estimate."""
    g = float(_bnp._flat(G0_A)[0])
    M = float(alpha)
    v = g * (1.0 - g) / (1.0 + M)
    res = RichResult(payload={"estimate": v,
                              "method": "DP variance (GvdV 2017 eq. 4.3)"})
    return with_describe_pointer(res, "gh_c4_3")


def cheatsheet():
    return "gh_c4_3: DP prior variance"
