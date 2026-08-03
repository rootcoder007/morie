# morie.fn -- function file (rootcoder007/morie)
"""Exponential-link density.

Implements sec. 2.3.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_exponential_link_density"]


def ghosal_ch2_exponential_link_density(f, x=0.5, mu=None,
                                        n_int=800):
    """p_f(x) = exp(f(x) - c(f)), c(f) = log int e^f dmu
    (sec. 2.3.1): normalizes on [0,1] with Lebesgue mu. ``f`` is a
    coefficient list on the cosine basis. Keys: distribution."""
    b = _bnp._flat(f)
    def fx(t):
        return sum(bj * math.sqrt(2.0)
                   * math.cos((j + 1) * math.pi * t)
                   for j, bj in enumerate(b))
    Z = sum(math.exp(fx((i + 0.5) / n_int))
            for i in range(n_int)) / n_int
    dens = math.exp(fx(float(x))) / Z
    res = RichResult(payload={"estimate": dens,
                              "distribution": dens,
                              "log_norm_const": math.log(Z),
                              "method": "exp-link density (GvdV 2017 sec. 2.3.1)"})
    return with_describe_pointer(res, "ghs004")


def cheatsheet():
    return "ghs004: Exponential-link density"
