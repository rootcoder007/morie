# morie.fn -- function file (rootcoder007/morie)
"""Series (eigenexpansion) GP.

Implements Example 11.4 + Example 11.16 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_series_gp"]


def ghosal_series_gp(x=0.3, y=0.7, n_terms=60):
    """K(x, y) = sum_k lambda_k phi_k(x) phi_k(y): a random series
    W = sum a_k(t) Z_k has exactly this covariance (Ex 11.4/11.16).
    Cosine eigenbasis with lambda_k = k^{-2}; Mercer partial sums
    converge. Keys: estimate."""
    def phi(t, k):
        return math.sqrt(2.0) * math.cos(k * math.pi * t)
    partial = []
    tot = 0.0
    for k in range(1, n_terms + 1):
        tot += k ** (-2.0) * phi(x, k) * phi(y, k)
        if k in (5, 20, n_terms):
            partial.append(tot)
    res = RichResult(payload={"estimate": tot,
                              "partial_sums": partial,
                              "converging": abs(partial[-1]
                                                - partial[-2])
                              < abs(partial[-2] - partial[-3])
                              + 1e-12,
                              "method": "eigenexpansion GP kernel (GvdV 2017 Ex 11.16)"})
    return with_describe_pointer(res, "gh_c11_10")


def cheatsheet():
    return "gh_c11_10: Series (eigenexpansion) GP"


# compact alias per ledger/NAMING.md
ghosalseriesgp = ghosal_series_gp
