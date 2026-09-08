# morie.fn -- function file (rootcoder007/morie)
"""GP binary-regression contraction.

Implements Theorem 11.22 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_binreg_crt"]


def ghosal_gp_binreg_crt(s=2.0, d=1.0, ns=(100, 10000)):
    """p(x) = Psi(W_x): if phi_{w0}(eps) ~ eps^{-d/s} (an s-smooth
    GP on [0,1]^d) the rate equation gives eps_n = n^{-s/(2s+d)}
    (Thm 11.22 + Thm 11.20). Keys: estimate."""
    s = float(s)
    d = float(d)
    rates = [float(n) ** (-s / (2.0 * s + d)) for n in ns]
    res = RichResult(payload={"estimate": rates[-1],
                              "rate_by_n": rates,
                              "exponent": s / (2.0 * s + d),
                              "method": "GP binary regression rate (GvdV 2017 Thm 11.22)"})
    return with_describe_pointer(res, "gh_c11_5")


def cheatsheet():
    return "gh_c11_5: GP binary-regression contraction"
