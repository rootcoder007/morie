# morie.fn -- function file (rootcoder007/morie)
"""Countable additivity of tree measures.

Implements sec. 3.6 (atom products vanish, p.43) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tree_countable_additivity"]


def ghosal_ch3_tree_countable_additivity(EV, depth=60):
    """E[V_e V_{e0} V_{e00} ...] = 0: with E(V) bounded away from 1
    the expected infinite down-branch product vanishes -- atoms have
    zero mass (GvdV 2017 p.43). Returns prod_{j<=depth} E(V), which
    decays geometrically. Keys: value."""
    ev = float(_bnp._flat(EV)[0])
    if not 0.0 <= ev < 1.0:
        raise ValueError("E(V) must lie in [0, 1)")
    prod = ev ** int(depth)
    res = RichResult(payload={"estimate": prod, "value": prod,
                              "vanishes": prod < 1e-12,
                              "method": "atom mass product (GvdV 2017 sec. 3.6)"})
    return with_describe_pointer(res, "ghs020")


def cheatsheet():
    return "ghs020: Countable additivity of tree measures"
