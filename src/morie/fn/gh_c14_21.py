# morie.fn -- function file (rootcoder007/morie)
"""Ordering-dependent stick-breaking.

Implements sec. 14.9.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ord_dep_sbp"]


def ghosal_ord_dep_sbp(x=0.3, atom_locs=(0.1, 0.35, 0.6, 0.9),
                       V=(0.5, 0.5, 0.5, 0.5)):
    """V_k applied in the order of proximity to x: closer atoms take
    their stick first, so nearby atoms dominate the local measure
    (sec. 14.9.4). Keys: estimate."""
    locs = _bnp._flat(atom_locs)
    Vs = _bnp._flat(V)
    order = sorted(range(len(locs)), key=lambda i: abs(locs[i] - x))
    W = [0.0] * len(locs)
    left = 1.0
    for rank, i in enumerate(order):
        W[i] = left * Vs[rank]
        left *= (1.0 - Vs[rank])
    nearest = order[0]
    res = RichResult(payload={"estimate": W[nearest],
                              "weights": W,
                              "nearest_dominates":
                                  W[nearest] == max(W),
                              "method": "ordering-dependent sticks (GvdV 2017 sec. 14.9.4)"})
    return with_describe_pointer(res, "gh_c14_21")


def cheatsheet():
    return "gh_c14_21: Ordering-dependent stick-breaking"
