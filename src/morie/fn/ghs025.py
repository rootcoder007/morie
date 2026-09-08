# morie.fn -- function file (rootcoder007/morie)
"""Tail-free density as branch product.

Implements Theorem 3.16, eq. (3.18), p.44 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_tailfree_density_product"]


def ghosal_ch3_tailfree_density_product(V_path):
    """p(x) = prod_j (2 V_{x_1..x_j}) (eq. 3.18): the density of a
    tail-free measure with respect to the canonical measure.
    Keys: distribution."""
    vs = _bnp._flat(V_path)
    p = 1.0
    for v in vs:
        p *= 2.0 * v
    res = RichResult(payload={"estimate": p, "distribution": p,
                              "depth": len(vs),
                              "method": "density product (GvdV 2017 eq. 3.18)"})
    return with_describe_pointer(res, "ghs025")


def cheatsheet():
    return "ghs025: Tail-free density as branch product"
