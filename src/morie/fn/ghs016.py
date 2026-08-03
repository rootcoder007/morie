# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet posterior covariance.

Implements the finite-Dirichlet form heading eq. (3.7), pp.32-33 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_dirichlet_posterior_cov"]


def ghosal_ch3_dirichlet_posterior_cov(alpha, counts, j, jp,
                                       alpha_total):
    """cov(p_j, p_j' | X) = -m_j m_j' / (A + n + 1). Keys: value."""
    v = _bnp.cdp_posterior_cov(alpha, counts, int(j), int(jp),
                               alpha_total)
    res = RichResult(payload={"estimate": v, "value": v,
                              "method": "Dirichlet posterior covariance (GvdV 2017 sec. 3.3.3)"})
    return with_describe_pointer(res, "ghs016")


def cheatsheet():
    return "ghs016: Dirichlet posterior covariance"
