# morie.fn -- function file (rootcoder007/morie)
"""Multinomial logistic probabilities.

Implements eq. (7.6) p.225 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(X, beta0, beta, baseline_last=True):
    """P(Y_i = c | x_i) = exp(beta_0c + x_i'beta_c)
    / sum_l exp(beta_0l + x_i'beta_l) (eq. 7.6).  The model is not
    identifiable as written, so the book sets
    (beta_0C, beta_C) = (0, 0) for the baseline category (p.225),
    which is what ``baseline_last`` does. Keys: estimate."""
    P = _gp.multinomial_probabilities(X, beta0, beta,
                                      baseline_last=baseline_last)
    res = RichResult(payload={"estimate": P[0][0],
                              "probabilities": P,
                              "n_categories": len(P[0]),
                              "method": "multinomial logistic probabilities (MVSML 2022 eq. 7.6)"})
    return with_describe_pointer(res, "msm120")


def cheatsheet():
    return "msm120: Multinomial logistic probabilities"
