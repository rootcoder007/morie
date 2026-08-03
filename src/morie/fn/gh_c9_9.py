# morie.fn -- function file (rootcoder007/morie)
"""White-noise conjugate posterior.

Implements sec. 9.5.4 (Ex 8.6, eq. 8.1 exact form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wn_conj_crt"]


def ghosal_wn_conj_crt(X, n, prior_var):
    """dY = theta dt + dW/sqrt(n) with theta_i ~ N(0, lambda_i):
    theta_i | Y ~ N(n X_i/(n + 1/lambda_i), 1/(n + 1/lambda_i)) --
    the exact coordinatewise conjugate posterior (eq. 8.1 form,
    sec. 9.5.4). Keys: estimate."""
    xs = _bnp._flat(X)
    lam = _bnp._flat(prior_var)
    n = float(n)
    means = [n * x / (n + 1.0 / l) for x, l in zip(xs, lam)]
    vars_ = [1.0 / (n + 1.0 / l) for l in lam]
    res = RichResult(payload={"estimate": means[0],
                              "posterior_mean": means,
                              "posterior_var": vars_,
                              "method": "white-noise conjugate posterior (GvdV 2017 sec. 9.5.4, eq. 8.1)"})
    return with_describe_pointer(res, "gh_c9_9")


def cheatsheet():
    return "gh_c9_9: White-noise conjugate posterior"
