# morie.fn -- function file (rootcoder007/morie)
"""Normal-mixture approximation.

Implements sec. 9.4.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_norm_mix_apx"]


def ghosal_norm_mix_apx(sigmas=(0.5, 0.25, 0.125), n_int=800):
    """Smooth p0 admits ||p0 - phi_sigma * G||_1 -> 0 as sigma -> 0
    (sec. 9.4.1): with G = P0 itself the convolution bias decays like
    sigma^2 for twice-smooth p0. Computes the L1 gap for p0 the
    Beta(2,2)-type density 6x(1-x). Keys: estimate."""
    def p0(x):
        return 6.0 * x * (1.0 - x) if 0.0 <= x <= 1.0 else 0.0
    def npdf(x, m, s):
        z = (x - m) / s
        return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))
    gaps = []
    for s in sigmas:
        gap = 0.0
        for i in range(n_int):
            x = -0.5 + 2.0 * (i + 0.5) / n_int
            conv = sum(p0((j + 0.5) / 200) * npdf(x, (j + 0.5) / 200,
                                                  s)
                       for j in range(200)) / 200
            gap += abs(conv - p0(x)) * 2.0 / n_int
        gaps.append(gap)
    res = RichResult(payload={"estimate": gaps[-1],
                              "l1_gap_by_sigma": gaps,
                              "improving": gaps[-1] < gaps[0],
                              "method": "normal-mixture approximation (GvdV 2017 sec. 9.4.1)"})
    return with_describe_pointer(res, "gh_c9_5")


def cheatsheet():
    return "gh_c9_5: Normal-mixture approximation"
