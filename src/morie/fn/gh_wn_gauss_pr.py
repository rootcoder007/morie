# morie.fn -- function file (rootcoder007/morie)
"""White-noise GP posterior.

Implements sec. 9.5.4 (conjugate Gaussian computation) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_white_noise_gauss_prior"]


def ghosal_white_noise_gauss_prior(Y, n, prior_sd):
    """dY = theta dt + dW/sqrt(n), theta ~ GP(0, C) diagonal in the
    basis: theta | Y ~ N(C(C + I/n)^{-1} Y, (C^{-1} + n I)^{-1})
    coordinatewise (sec. 9.5.4). Keys: estimate."""
    ys = _bnp._flat(Y)
    sds = _bnp._flat(prior_sd)
    n = float(n)
    means = []
    vars_ = []
    for y, sd in zip(ys, sds):
        c = sd * sd
        means.append(c / (c + 1.0 / n) * y)
        vars_.append(1.0 / (1.0 / c + n))
    res = RichResult(payload={"estimate": means[0],
                              "posterior_mean": means,
                              "posterior_var": vars_,
                              "shrinkage": [c and m / y
                                            for m, y, c in
                                            zip(means, ys,
                                                [1] * len(ys))
                                            if y != 0],
                              "method": "white-noise GP posterior (GvdV 2017 sec. 9.5.4)"})
    return with_describe_pointer(res, "gh_wn_gauss_pr")


def cheatsheet():
    return "gh_wn_gauss_pr: White-noise GP posterior"
