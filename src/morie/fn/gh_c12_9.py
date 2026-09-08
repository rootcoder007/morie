# morie.fn -- function file (rootcoder007/morie)
"""White-noise full BvM.

Implements sec. 12.4.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wn_full_bvm"]


def ghosal_wn_full_bvm(y=(0.5, -0.2, 0.1), n=400, prior_var=1e6):
    """dY = theta dt + dW/sqrt(n): with a (nearly) flat Gaussian
    prior the posterior is N(Y, I/n) exactly -- full BvM in total
    variation (sec. 12.4.1). Exact conjugate computation.
    Keys: estimate."""
    ys = _bnp._flat(y)
    n = float(n)
    means = [prior_var / (prior_var + 1.0 / n) * v for v in ys]
    vars_ = [1.0 / (1.0 / prior_var + n) for _ in ys]
    gap = max(abs(m - v) for m, v in zip(means, ys))
    var_gap = max(abs(v - 1.0 / n) for v in vars_)
    res = RichResult(payload={"estimate": gap,
                              "mean_matches_Y": gap < 1e-6,
                              "var_matches_In": var_gap < 1e-8,
                              "method": "white-noise full BvM (GvdV 2017 sec. 12.4.1)"})
    return with_describe_pointer(res, "gh_c12_9")


def cheatsheet():
    return "gh_c12_9: White-noise full BvM"
