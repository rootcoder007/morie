# morie.fn -- function file (rootcoder007/morie)
"""Number of distinct values.

Implements Proposition 4.8 (exact mean/variance + bounds) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_ndist"]


def ghosal_dp_ndist(n, alpha):
    """E K_n = sum_{i=1}^n M/(M+i-1); var K_n =
    sum M(i-1)/(M+i-1)^2; and the bounds 1 v M log(1+n/M) <= E K_n
    <= 1 + M log(1+n/M) (Prop 4.8). Keys: estimate."""
    M = float(alpha)
    n = int(n)
    mean = sum(M / (M + i - 1.0) for i in range(1, n + 1))
    var = sum(M * (i - 1.0) / (M + i - 1.0) ** 2
              for i in range(1, n + 1))
    lo = max(1.0, M * math.log1p(n / M))
    hi = 1.0 + M * math.log1p(n / M)
    res = RichResult(payload={"estimate": mean, "variance": var,
                              "lower_bound": lo, "upper_bound": hi,
                              "bounds_hold": lo <= mean <= hi,
                              "method": "distinct-value moments (GvdV 2017 Prop 4.8)"})
    return with_describe_pointer(res, "gh_c4_8")


def cheatsheet():
    return "gh_c4_8: Number of distinct values"
