# morie.fn -- function file (rootcoder007/morie)
"""Set probability as branch product.

Implements eq. (3.12), sec. 3.6, p.39 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tree_set_probability"]


def ghosal_ch3_tree_set_probability(V_path, epsilon=None):
    """P(A_{e_1..e_m}) = V_{e_1} V_{e_1 e_2} ... V_{e_1..e_m}: the
    product of splitting variables down the branch. Keys: value."""
    vs = _bnp._flat(V_path)
    p = 1.0
    for v in vs:
        if not 0.0 <= v <= 1.0:
            raise ValueError("splitting variables must lie in [0,1]")
        p *= v
    res = RichResult(payload={"estimate": p, "value": p,
                              "depth": len(vs),
                              "method": "branch product (GvdV 2017 sec. 3.6)"})
    return with_describe_pointer(res, "ghs019")


def cheatsheet():
    return "ghs019: Set probability as branch product"
