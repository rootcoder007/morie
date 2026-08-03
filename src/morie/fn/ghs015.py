# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet posterior variance.

Implements the finite-Dirichlet form heading eq. (3.7), pp.32-33 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_dirichlet_posterior_var"]


def ghosal_ch3_dirichlet_posterior_var(alpha, counts, j, alpha_total):
    """var(p_j | X) = (alpha_j+N_j)(A + n - alpha_j - N_j) /
    ((A+n)^2 (A+n+1)) = m_j (1-m_j)/(A+n+1). Keys: value."""
    v = _bnp.cdp_posterior_var(alpha, counts, int(j), alpha_total)
    res = RichResult(payload={"estimate": v, "value": v,
                              "method": "Dirichlet posterior variance (GvdV 2017 sec. 3.3.3)"})
    return with_describe_pointer(res, "ghs015")


def cheatsheet():
    return "ghs015: Dirichlet posterior variance"
