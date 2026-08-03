# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric normal regression with a GP prior.

Implements sec. 2.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_np_normal_reg"]


def ghosal_np_normal_reg(x, y, length=0.5, var=1.0, sigma2=0.05):
    """Y_i = f(x_i) + e_i, e ~ N(0, sigma^2), f ~ GP(0, k) (GvdV 2017
    sec. 2.4): the posterior mean is the kernel-ridge smoother
    k(x*, X)[K + sigma^2 I]^{-1} Y."""
    xs = _bnp._flat(x)
    ys = _bnp._flat(y)
    k = _bnp.rbf_kernel(length, var)
    fhat = _bnp.gp_regression_posterior_mean(xs, ys, xs, k, sigma2)
    sse = sum((a - b) ** 2 for a, b in zip(fhat, ys))
    res = RichResult(payload={"estimate": sum(fhat) / len(fhat),
                              "fitted": fhat, "sse": sse,
                              "method": "GP-prior normal regression (GvdV 2017 sec. 2.4)"})
    return with_describe_pointer(res, "gh_c2_8")


def cheatsheet():
    return "gh_c2_8: Nonparametric normal regression with a GP prior"
