# morie.fn -- function file (rootcoder007/morie)
"""Feller density approximation.

Implements sec. 2.3.4 (Feller operator) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_feller_density_approximation"]


def ghosal_ch2_feller_density_approximation(x=0.4, k=40, F=None,
                                            h_k=None, g_k=None,
                                            V=None):
    """a(x; k, F) = int h_k(x; z) dF(z) (sec. 2.3.4): with the
    binomial Feller kernel this is the derivative-scale Bernstein
    operator k * sum_j [F((j+1)/k) - F(j/k)] b_{j,k-1}(x), which
    approximates the density F'. Default truth F(z) = z^2 (density
    2z). Keys: value."""
    if F is None:
        F = lambda z: max(0.0, min(1.0, z)) ** 2
    k = int(k)
    dens = 0.0
    for j in range(k):
        inc = F((j + 1.0) / k) - F(j / k)
        b = math.comb(k - 1, j) * x ** j * (1.0 - x) ** (k - 1 - j)
        dens += k * inc * b
    truth = 2.0 * x
    res = RichResult(payload={"estimate": dens, "value": dens,
                              "truth": truth,
                              "gap": abs(dens - truth),
                              "method": "Feller density approximation (GvdV 2017 sec. 2.3.4)"})
    return with_describe_pointer(res, "ghs006")


def cheatsheet():
    return "ghs006: Feller density approximation"
