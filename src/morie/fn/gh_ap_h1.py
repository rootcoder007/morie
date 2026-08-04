# morie.fn -- function file (rootcoder007/morie)
"""Inverse-Gaussian density.

Implements Appendix H of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_inv_gauss"]


def ghosal_inv_gauss(x=1.0, alpha_loc=1.0, gamma_sh=2.0,
                     n_grid=6000, x_max=12.0):
    """f(x; a, g) = sqrt(g/(2 pi x^3)) exp(-g (x-a)^2/(2 a^2 x))
    (App H): density evaluated + quadrature check of unit mass.
    Keys: estimate."""
    def pdf(v):
        return math.sqrt(gamma_sh / (2.0 * math.pi * v ** 3)) \
            * math.exp(-gamma_sh * (v - alpha_loc) ** 2
                       / (2.0 * alpha_loc ** 2 * v))
    val = pdf(x)
    mass = sum(pdf((i + 0.5) * x_max / n_grid) * x_max / n_grid
               for i in range(n_grid))
    res = RichResult(payload={"estimate": val,
                              "total_mass": mass,
                              "normalized": abs(mass - 1.0) < 5e-3,
                              "method": "inverse-Gaussian density (GvdV 2017 App H)"})
    return with_describe_pointer(res, "gh_ap_h1")


def cheatsheet():
    return "gh_ap_h1: Inverse-Gaussian density"


# compact alias per ledger/NAMING.md
ghosalinvgauss = ghosal_inv_gauss
