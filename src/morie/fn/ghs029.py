# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree density moments.

Implements eq. (3.22), p.49 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_polya_tree_density_moments"]


def ghosal_ch3_polya_tree_density_moments(alpha_path, depth=None):
    """E p(x) = prod_j 2 alpha_{x_1..x_j} / (alpha_..0 + alpha_..1)
    and E p(x)^2 = prod_j 4 alpha (alpha+1) / ((a0+a1)(a0+a1+1))
    (eq. 3.22). ``alpha_path`` is (alpha_taken, alpha_other) pairs.
    Keys: value."""
    pairs = [(float(a), float(b)) for a, b in alpha_path]
    if depth is not None:
        pairs = pairs[:int(depth)]
    m1 = 1.0
    m2 = 1.0
    for a_take, a_other in pairs:
        s = a_take + a_other
        m1 *= 2.0 * a_take / s
        m2 *= 4.0 * a_take * (a_take + 1.0) / (s * (s + 1.0))
    res = RichResult(payload={"estimate": m1, "value": [m1, m2],
                              "second_moment": m2,
                              "method": "PT density moments (GvdV 2017 eq. 3.22)"})
    return with_describe_pointer(res, "ghs029")


def cheatsheet():
    return "ghs029: Pólya tree density moments"
