# morie.fn -- function file (rootcoder007/morie)
"""Location-scale mixture limit.

Implements sec. 2.3.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch2_location_scale_mixture_limit"]


def ghosal_ch2_location_scale_mixture_limit(psi=None, f=None,
                                            sigma=(0.5, 0.1, 0.02),
                                            mu=None, x=0.5,
                                            n_int=400):
    """int sigma^{-1} psi((x - mu)/sigma) f(mu) dmu -> f(x) as
    sigma -> 0 (sec. 2.3.3): the kernel-convolution bias vanishes.
    Default: normal psi, f = 6x(1-x). Keys: value."""
    if f is None:
        f = lambda t: 6.0 * t * (1.0 - t) if 0 <= t <= 1 else 0.0
    if psi is None:
        psi = lambda z: math.exp(-0.5 * z * z) \
            / math.sqrt(2.0 * math.pi)
    gaps = []
    for s in _bnp._flat(sigma):
        conv = sum(psi((x - (i + 0.5) / n_int) / s) / s
                   * f((i + 0.5) / n_int) for i in range(n_int)) \
            / n_int
        gaps.append(abs(conv - f(x)))
    res = RichResult(payload={"estimate": gaps[-1], "value": gaps,
                              "converging": gaps[-1] < gaps[0],
                              "method": "location-scale mixture limit (GvdV 2017 sec. 2.3.3)"})
    return with_describe_pointer(res, "ghs005")


def cheatsheet():
    return "ghs005: Location-scale mixture limit"
