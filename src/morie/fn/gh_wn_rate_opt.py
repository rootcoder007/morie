# morie.fn -- function file (rootcoder007/morie)
"""White-noise minimax rate.

Implements sec. 8.3.4 context of Ex 8.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_white_noise_optimal_rate"]


def ghosal_white_noise_optimal_rate(smoothness, n):
    """Minimax squared-error rate in the white-noise model over a
    Sobolev-s ball: R_n* = n^{-2s/(2s+1)} (sec. 8.3.4); attained by
    the conjugate prior with alpha = s (Ex 8.6). Keys: estimate."""
    s = float(smoothness)
    rate = float(n) ** (-2.0 * s / (2.0 * s + 1.0))
    res = RichResult(payload={"estimate": rate,
                              "exponent": 2.0 * s / (2.0 * s + 1.0),
                              "attained_by_alpha_eq_s": True,
                              "method": "white-noise minimax rate (GvdV 2017 sec. 8.3.4)"})
    return with_describe_pointer(res, "gh_wn_rate_opt")


def cheatsheet():
    return "gh_wn_rate_opt: White-noise minimax rate"
