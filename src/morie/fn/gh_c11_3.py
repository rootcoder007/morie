# morie.fn -- function file (rootcoder007/morie)
"""Gaussian contraction-rate equation.

Implements Theorem 11.20, eq. (11.12) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_crt_thm"]


def ghosal_gp_crt_thm(phi_exponent=2.0, n=10000):
    """The rate solves phi_{w0}(eps_n) <= n eps_n^2 (eq. 11.12).
    With phi(eps) ~ eps^{-a} (Brownian motion: a = 2, Lemma 11.27)
    the minimal solution is eps_n = n^{-1/(2+a)} -- n^{-1/4} for BM.
    Keys: estimate."""
    a = float(phi_exponent)
    eps_n = float(n) ** (-1.0 / (2.0 + a))
    gap = abs(eps_n ** (-a) - float(n) * eps_n ** 2) \
        / (float(n) * eps_n ** 2)
    res = RichResult(payload={"estimate": eps_n,
                              "rate_exponent": 1.0 / (2.0 + a),
                              "balance_gap": gap,
                              "method": "Gaussian rate equation (GvdV 2017 Thm 11.20)"})
    return with_describe_pointer(res, "gh_c11_3")


def cheatsheet():
    return "gh_c11_3: Gaussian contraction-rate equation"
