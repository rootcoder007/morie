# morie.fn -- function file (rootcoder007/morie)
"""Normalized completely random measure.

Implements sec. 14.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ncrm_def"]


def ghosal_ncrm_def(jump_sizes, jump_locs, set_lo=0.0, set_hi=0.5):
    """G(A) = M(A)/M(X) with M(A) = sum_{tau_k in A} J_k
    (sec. 14.7): evaluates the normalized measure of a set.
    Keys: estimate."""
    J = _bnp._flat(jump_sizes)
    tau = _bnp._flat(jump_locs)
    T = sum(J)
    if T <= 0:
        raise ValueError("total jump mass must be positive")
    mass = sum(j for j, t in zip(J, tau) if set_lo <= t < set_hi) / T
    res = RichResult(payload={"estimate": mass,
                              "total_mass": 1.0,
                              "method": "NCRM set mass (GvdV 2017 sec. 14.7)"})
    return with_describe_pointer(res, "gh_c14_15")


def cheatsheet():
    return "gh_c14_15: Normalized completely random measure"


# compact alias per ledger/NAMING.md
ghosalncrmdef = ghosal_ncrm_def
