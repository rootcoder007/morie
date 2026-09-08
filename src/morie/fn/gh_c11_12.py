# morie.fn -- function file (rootcoder007/morie)
"""Self-similarity of fBm.

Implements sec. 11.5.1 (Ex 11.5/11.9 self-similarity) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_selfsim_gp"]


def ghosal_selfsim_gp(H=0.6, lam=3.0, t=0.2):
    """fBm: f(lambda t) =_d lambda^H f(t) -- variances satisfy
    E f(lambda t)^2 = lambda^{2H} E f(t)^2 exactly, from the kernel
    (11.6) (sec. 11.5.1). Keys: estimate."""
    H = float(H)
    v1 = t ** (2 * H)
    v2 = (lam * t) ** (2 * H)
    ratio = v2 / v1
    res = RichResult(payload={"estimate": ratio,
                              "expected": lam ** (2 * H),
                              "gap": abs(ratio - lam ** (2 * H)),
                              "method": "fBm self-similarity (GvdV 2017 sec. 11.5.1)"})
    return with_describe_pointer(res, "gh_c11_12")


def cheatsheet():
    return "gh_c11_12: Self-similarity of fBm"
