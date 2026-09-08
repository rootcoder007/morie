# morie.fn -- function file (rootcoder007/morie)
"""Penalized Poisson regression.

Implements eq. (7.11) p.232 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["penalized_poisson_fit"]


def penalized_poisson_fit(X, y, lam=1.0, penalty="ridge", add_intercept=True):
    """P(Y_i = y | x_i) = lambda_i^y exp(-lambda_i)/y! with
    lambda_i = exp(beta_0 + x_i'beta_0) (eq. 7.11).  Estimated by the
    penalized likelihood of p.232,
    l_p = sum_i y_i eta_i - sum_i exp(eta_i) - sum_i log(y_i!)
    - (lambda/2) sum_j beta_j^2, solved by iteratively reweighted
    least squares; ``penalty='lasso'`` switches to the L1 version.
    Keys: estimate."""
    f = _gp.penalized_poisson_fit(X, y, lam=lam, penalty=penalty,
                                  add_intercept=add_intercept)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"],
                              "fitted": f["fitted"],
                              "loglik": f["loglik"],
                              "penalized_loglik": f["penalized_loglik"],
                              "iterations": f["iterations"],
                              "method": "penalized Poisson regression (MVSML 2022 eq. 7.11)"})
    return with_describe_pointer(res, "msm122")


def cheatsheet():
    return "msm122: Penalized Poisson regression"
