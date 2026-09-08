# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric efficiency bound.

Implements sec. 12.3 (Cramer-Rao form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_semipara_eff"]


def ghosal_semipara_eff(grad_psi, info_matrix):
    """var >= (nabla psi)' I^{-1} (nabla psi): the semiparametric
    Cramer-Rao lower bound (sec. 12.3). Solves I x = grad and forms
    the quadratic. Keys: estimate."""
    g = _bnp._flat(grad_psi)
    I = [[float(v) for v in row] for row in info_matrix]
    x = np.linalg.solve(np.marr(I), np.marr(g))
    xl = [float(v) for v in x._flat()]
    bound = sum(a * b for a, b in zip(g, xl))
    res = RichResult(payload={"estimate": bound,
                              "positive": bound > 0,
                              "method": "semiparametric efficiency bound (GvdV 2017 sec. 12.3)"})
    return with_describe_pointer(res, "gh_c12_6")


def cheatsheet():
    return "gh_c12_6: Semiparametric efficiency bound"
