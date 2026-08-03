# morie.fn -- function file (rootcoder007/morie)
"""Penalized multinomial log-likelihood (ridge).

Implements eq. (7.7) p.226 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_7"]


def mvsml_bayesian_regression_pt2_eq_7_7(X, y, beta0, beta, lam=1.0, baseline_last=True):
    """l_p(beta; y) = l(beta; y) - lambda sum_c beta_c'beta_c
    (eq. 7.7): the quadratic-regularized multinomial likelihood, which
    removes the need for the identifiability constraint on the slopes
    (p.226).  Intercepts are never penalized. Keys: estimate."""
    f = _gp.penalized_multinomial_loglik(X, y, beta0, beta, lam,
                                         penalty="ridge",
                                         baseline_last=baseline_last)
    res = RichResult(payload={"estimate": f["penalized_loglik"],
                              "loglik": f["loglik"],
                              "penalty": f["penalty"],
                              "method": "ridge-penalized multinomial log-likelihood (MVSML 2022 eq. 7.7)"})
    return with_describe_pointer(res, "msm111")


def cheatsheet():
    return "msm111: Penalized multinomial log-likelihood (ridge)"
