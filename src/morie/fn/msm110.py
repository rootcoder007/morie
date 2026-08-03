# morie.fn -- function file (rootcoder007/morie)
"""Multinomial log-likelihood.

Implements eq. (7.8) p.226 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_8"]


def mvsml_bayesian_regression_pt2_eq_7_8(X, y, beta0, beta, baseline_last=True):
    """l(beta; y) = sum_i sum_c 1{y_i = c}(beta_0c + x_i'beta_c)
    - sum_i log[sum_l exp(beta_0l + x_i'beta_l)] (eq. 7.8).
    Keys: estimate."""
    ll = _gp.multinomial_loglik(X, y, beta0, beta,
                                baseline_last=baseline_last)
    res = RichResult(payload={"estimate": ll, "loglik": ll,
                              "method": "multinomial log-likelihood (MVSML 2022 eq. 7.8)"})
    return with_describe_pointer(res, "msm110")


def cheatsheet():
    return "msm110: Multinomial log-likelihood"
