# morie.fn -- function file (rootcoder007/morie)
"""Bayesian ridge regression (BRR).

Implements eq. (6.3) pp.173-175 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def mvsml_bayesian_regression_eq_6_3(y, X, n_iter=2000, burn_in=500, nu=5.0, nu_beta=5.0,
         R2=0.5, seed=42):
    """Y = mu + sum_j X_j beta_j + eps (eq. 6.3) with a flat prior on
    mu, beta | sigma2_beta ~ N_p(0, I sigma2_beta) and scaled inverse
    chi-square priors on both variances.  Fitted with the six-step
    Gibbs sampler of pp.174-175; the BGLR hyperparameter defaults
    S = Var(Y)(1 - R2)(nu + 2), S_beta = Var(Y) R2 (nu_beta + 2) are
    used. Keys: estimate."""
    f = _gp.bayes_ridge_gibbs(y, X, n_iter=n_iter, burn_in=burn_in,
                              nu=nu, nu_beta=nu_beta, R2=R2,
                              seed=seed)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "beta": f["beta"],
                              "sigma2": f["sigma2"],
                              "sigma2_beta": f["sigma2_beta"],
                              "n_kept": f["n_kept"],
                              "method": "Bayesian ridge regression (MVSML 2022 eq. 6.3)"})
    return with_describe_pointer(res, "msm046")


def cheatsheet():
    return "msm046: Bayesian ridge regression (BRR)"
