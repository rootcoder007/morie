# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet moments.

Implements Appendix G (Corollary G.4 forms) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dir_moments"]


def ghosal_dir_moments(alpha, j=0, jp=1):
    """E X_j = alpha_j/A; cov(X_i, X_j) = -alpha_i alpha_j /
    (A^2 (A+1)) for i != j; var X_j = alpha_j(A - alpha_j)/(A^2(A+1))
    (App G). Keys: estimate."""
    a = _bnp._flat(alpha)
    A = sum(a)
    mean = a[j] / A
    var = a[j] * (A - a[j]) / (A * A * (A + 1.0))
    cov = -a[j] * a[jp] / (A * A * (A + 1.0))
    res = RichResult(payload={"estimate": mean,
                              "variance": var, "covariance": cov,
                              "method": "Dirichlet moments (GvdV 2017 App G)"})
    return with_describe_pointer(res, "gh_ap_g2")


def cheatsheet():
    return "gh_ap_g2: Dirichlet moments"
