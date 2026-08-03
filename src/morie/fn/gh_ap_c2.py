# morie.fn -- function file (rootcoder007/morie)
"""Packing numbers.

Implements Appendix C of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_packing_num"]


def ghosal_packing_num(radius_set=1.0, eps=0.25, dim=2):
    """D(eps, T, d) = max #{eps-separated points} (App C):
    N(eps) <= D(eps) <= N(eps/2). Keys: estimate."""
    N_eps = (3.0 * radius_set / eps) ** dim
    N_half = (6.0 * radius_set / eps) ** dim
    D = N_eps                        # a valid value in the bracket
    res = RichResult(payload={"estimate": D,
                              "sandwich": [N_eps, N_half],
                              "relation_holds": N_eps <= N_half,
                              "method": "packing numbers (GvdV 2017 App C)"})
    return with_describe_pointer(res, "gh_ap_c2")


def cheatsheet():
    return "gh_ap_c2: Packing numbers"
