# morie.fn -- function file (rootcoder007/morie)
"""Bayesian ordinal regression model.

Implements eq. (7.1) pp.210-213 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(y, X, n_iter=1200, burn_in=300, link="probit", seed=42,
         fit=True):
    """p_ic = P(Y_i = c) = F(gamma_c + x_i'beta)
    - F(gamma_{c-1} + x_i'beta) (eq. 7.1), the ordinal model built on
    a latent continuous variable cut at thresholds
    -inf = gamma_0 < gamma_1 < ... < gamma_C = +inf.  F is the
    standard normal CDF for the ordinal probit model and the standard
    logistic CDF for the ordinal logistic model (p.210).  With
    ``fit=True`` the probit model is estimated by the Albert and Chib
    Gibbs sampler of pp.212-213. Keys: estimate."""
    if not fit:
        eta = [sum(a * b for a, b in zip(row, _gp._flat(X[0])))
               for row in [[]]] if False else None
    f = _gp.ordinal_probit_gibbs(y, X, n_iter=n_iter,
                                 burn_in=burn_in, seed=seed)
    probs = _gp.ordinal_probabilities(
        _gp._mv(_gp._mat(X), f["beta"]), f["gamma"], link=link)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"],
                              "gamma": f["gamma"],
                              "sigma2_beta": f["sigma2_beta"],
                              "probabilities": probs,
                              "n_categories": f["n_categories"],
                              "method": "Bayesian ordinal regression (MVSML 2022 eq. 7.1)"})
    return with_describe_pointer(res, "msm091")


def cheatsheet():
    return "msm091: Bayesian ordinal regression model"
