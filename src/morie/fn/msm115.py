# morie.fn -- function file (rootcoder007/morie)
"""Lasso-penalized multinomial log-likelihood.

Implements eq. (7.10) p.227 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_10"]


def mvsml_bayesian_regression_pt2_eq_7_10(X, y, beta0, beta, lam=1.0, baseline_last=True):
    """l_p(beta; y) = l(beta; y) - lambda sum_c sum_j |beta_cj|
    (eq. 7.10): the same block updating as eq. (7.9) but with the
    quadratic penalty replaced by an L1 one. Keys: estimate."""
    f = _gp.penalized_multinomial_loglik(X, y, beta0, beta, lam,
                                         penalty="lasso",
                                         baseline_last=baseline_last)
    res = RichResult(payload={"estimate": f["penalized_loglik"],
                              "loglik": f["loglik"],
                              "penalty": f["penalty"],
                              "method": "lasso-penalized multinomial log-likelihood (MVSML 2022 eq. 7.10)"})
    return with_describe_pointer(res, "msm115")


def cheatsheet():
    return "msm115: Lasso-penalized multinomial log-likelihood"
