# morie.fn -- function file (rootcoder007/morie)
"""Dudley entropy bound.

Implements Appendix I of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dudley_entropy"]


def ghosal_dudley_entropy(sigma=1.0, entropy_exponent=1.0,
                          n_grid=2000):
    """E sup_f |G_n f| <= C int_0^sigma sqrt(log N(eps)) deps
    (App I): with log N ~ eps^{-a}, the integral is finite iff a < 2.
    Quadrature of the entropy integral. Keys: estimate."""
    a = float(entropy_exponent)
    J = 0.0
    for i in range(n_grid):
        eps = (i + 0.5) * sigma / n_grid
        J += math.sqrt(eps ** (-a)) * sigma / n_grid
    finite = a < 2.0
    res = RichResult(payload={"estimate": J,
                              "finite": finite,
                              "method": "Dudley entropy integral (GvdV 2017 App I)"})
    return with_describe_pointer(res, "gh_ap_i2")


def cheatsheet():
    return "gh_ap_i2: Dudley entropy bound"
