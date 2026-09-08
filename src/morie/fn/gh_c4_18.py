# morie.fn -- function file (rootcoder007/morie)
"""Distribution of the DP mean.

Implements Theorem 4.26 (Cifarelli-Regazzini) + Lemma 4.27 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_mean_dist"]


import cmath


def ghosal_dp_mean_dist(support, base_masses, s, t_max=200.0,
                        n_grid=20000):
    """H(s) = 1/2 - (1/pi) int_0^inf t^{-1}
    Im exp(-int log(1 + i t (s-x)) dbeta(x)) dt (Theorem 4.26), for
    beta a finite discrete base measure sum_j b_j delta_{x_j}; the
    inner integral is sum_j b_j log(1 + i t (s - x_j)). Keys:
    estimate."""
    xs = _bnp._flat(support)
    bs = _bnp._flat(base_masses)
    s = float(s)
    h = float(t_max) / n_grid
    acc = 0.0
    for i in range(n_grid):
        t = (i + 0.5) * h
        inner = sum(b * cmath.log(1.0 + 1j * t * (s - x))
                    for x, b in zip(xs, bs))
        acc += (cmath.exp(-inner)).imag / t * h
    H = 0.5 - acc / math.pi
    H = min(max(H, 0.0), 1.0)
    res = RichResult(payload={"estimate": H,
                              "method": "DP mean CDF, Cifarelli-Regazzini (GvdV 2017 Thm 4.26)"})
    return with_describe_pointer(res, "gh_c4_18")


def cheatsheet():
    return "gh_c4_18: Distribution of the DP mean"
