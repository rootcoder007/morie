# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree tail density bounds.

Implements sec. 3.7.2 (tail-factor bounds around eq. 3.23) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_polya_tree_density_bounds"]


def ghosal_ch3_polya_tree_density_bounds(n, a_of_level, m, depth):
    """prod_{j>m} (1 - n/(2 a_j)) <= prod_{j>m} (2 a_j + 2 N_j) /
    (2 a_j + N_{j-1}) <= prod_{j>m} (1 + n/a_j): with at most n
    observations on any path, each posterior factor beyond level m is
    sandwiched by these deterministic products (GvdV 2017
    sec. 3.7.2). Keys: value."""
    n = float(n)
    lo = 1.0
    hi = 1.0
    for j in range(int(m) + 1, int(depth) + 1):
        a = float(a_of_level(j))
        lo *= max(1.0 - n / (2.0 * a), 0.0)
        hi *= 1.0 + n / a
    res = RichResult(payload={"estimate": lo, "value": [lo, hi],
                              "lower": lo, "upper": hi,
                              "bracket_valid": lo <= 1.0 <= hi,
                              "method": "PT tail density bounds (GvdV 2017 sec. 3.7.2)"})
    return with_describe_pointer(res, "ghs032")


def cheatsheet():
    return "ghs032: Pólya tree tail density bounds"
