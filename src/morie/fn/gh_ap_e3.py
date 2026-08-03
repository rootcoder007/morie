# morie.fn -- function file (rootcoder007/morie)
"""Wavelet multiresolution expansion.

Implements Appendix E of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wavelet_mra"]


def ghosal_wavelet_mra(n_levels=6):
    """f = sum_k a_k phi_{j0,k} + sum_{j>=j0} sum_k d_jk psi_jk
    (App E): Haar MRA of f(x) = x on [0,1); the partial sums converge
    in L2 with detail energy 2^{-2j}/48 per level. Keys: estimate."""
    # Haar expansion of f(x)=x: father coefficient = 1/2;
    # details d_{j,k} = -2^{-3j/2 - 2}... check energy identity:
    total = 1.0 / 3.0                      # ||f||^2
    energy = 0.25                          # (int f phi)^2 with phi=1
    for j in range(n_levels):
        # sum_k d_{jk}^2 = 2^{-2j} / 16 * ... for Haar-of-x:
        # per-level detail energy = 2^{-2j}/48 * 4 = 2^{-2j}/12? use
        # exact: variance halving -- compute numerically
        lvl = 0.0
        for k in range(2 ** j):
            # d = int f psi_{jk}: psi = 2^{j/2} on first half, -2^{j/2}
            # on second half of [k2^-j, (k+1)2^-j]
            h = 2.0 ** (-j - 1)
            x1 = k * 2.0 ** (-j) + h / 2.0    # midpoints
            x2 = x1 + h
            d = 2.0 ** (j / 2.0) * (x1 * h - x2 * h)
            lvl += d * d
        energy += lvl
    res = RichResult(payload={"estimate": energy,
                              "l2_norm2": total,
                              "parseval_gap": abs(energy - total),
                              "method": "Haar MRA of f(x)=x (GvdV 2017 App E)"})
    return with_describe_pointer(res, "gh_ap_e3")


def cheatsheet():
    return "gh_ap_e3: Wavelet multiresolution expansion"
