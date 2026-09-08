# morie.fn -- function file (rootcoder007/morie)
"""Binary regression density.

Implements sec. 2.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_binary_regression_density"]


def ghosal_ch2_binary_regression_density(y, x=None, f=0.7, H=None):
    """p_f(y | x) = H(f(x))^y (1 - H(f(x)))^{1-y} (sec. 2.5): the
    likelihood of a link-transformed regression function. Default
    logistic H; ``f`` a scalar value f(x). Keys: distribution."""
    if H is None:
        H = lambda v: 1.0 / (1.0 + math.exp(-v))
    p = H(float(f))
    ys = _bnp._flat(y)
    lik = 1.0
    for yi in ys:
        lik *= p if yi > 0 else (1.0 - p)
    res = RichResult(payload={"estimate": lik,
                              "distribution": lik,
                              "success_prob": p,
                              "method": "binary regression density (GvdV 2017 sec. 2.5)"})
    return with_describe_pointer(res, "ghs007")


def cheatsheet():
    return "ghs007: Binary regression density"
