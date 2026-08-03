# morie.fn -- function file (rootcoder007/morie)
"""Basis truncation error.

Implements sec. 2.2 (approximation bound) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_basis_truncation_error"]


def ghosal_ch2_basis_truncation_error(f=None, J=8, alpha=2.0, k=1.0):
    """||f - sum_{j<=J} f_j psi_j|| <~ J^{-alpha/k} ||f||_alpha
    (sec. 2.2): computes the exact L2 truncation error of the cosine
    expansion for the default smooth f(x) = x(1-x) and the bound.
    Keys: value."""
    J = int(J)
    n_int = 800
    xs = [(i + 0.5) / n_int for i in range(n_int)]
    if f is None:
        f = lambda x: x * (1.0 - x)
    coefs = []
    for j in range(1, J + 1):
        c = sum(f(x) * math.sqrt(2.0) * math.cos(j * math.pi * x)
                for x in xs) / n_int
        coefs.append(c)
    mean = sum(f(x) for x in xs) / n_int
    err2 = 0.0
    for x in xs:
        approx = mean + sum(
            c * math.sqrt(2.0) * math.cos((j + 1) * math.pi * x)
            for j, c in enumerate(coefs))
        err2 += (f(x) - approx) ** 2 / n_int
    err = math.sqrt(err2)
    bound = float(J) ** (-alpha / k)
    res = RichResult(payload={"estimate": err, "value": err,
                              "bound_order": bound,
                              "within_order": err <= 10.0 * bound,
                              "method": "truncation error (GvdV 2017 sec. 2.2)"})
    return with_describe_pointer(res, "ghs003")


def cheatsheet():
    return "ghs003: Basis truncation error"
