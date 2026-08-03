# morie.fn -- function file (rootcoder007/morie)
"""Rescaled Gaussian process.

Implements Example 11.11 + sec. 11.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_rescal_gp"]


def ghosal_rescal_gp(lengths=(2.0, 1.0, 0.25), h=0.3):
    """f_l(x) = f(x / l): rescaling changes the kernel to
    K(s/l, t/l) (Ex 11.11) -- shrinking l roughens the paths; with a
    prior on l the smoothness adapts (sec. 11.5). Square-exponential:
    correlation at lag h is e^{-(h/l)^2}, decreasing as l shrinks.
    Keys: estimate."""
    cors = [math.exp(-(h / l) ** 2) for l in lengths]
    res = RichResult(payload={"estimate": cors[-1],
                              "correlation_by_length": cors,
                              "roughens_as_l_shrinks": all(
                                  cors[i + 1] <= cors[i] + 1e-15
                                  for i in range(len(cors) - 1)),
                              "method": "rescaled GP (GvdV 2017 Ex 11.11, sec. 11.5)"})
    return with_describe_pointer(res, "gh_c11_11")


def cheatsheet():
    return "gh_c11_11: Rescaled Gaussian process"
