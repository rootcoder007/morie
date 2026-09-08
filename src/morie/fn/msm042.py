# morie.fn -- function file (rootcoder007/morie)
"""Normal linear regression under the Bayesian paradigm.

Implements eq. (6.1)-(6.2) pp.172 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(X, y, add_intercept=True):
    """Y = beta_0 + sum_j X_j beta_j + eps (eq. 6.1) with the
    improper reference prior f(beta, sigma2) proportional to
    sigma^-2 (eq. 6.2).  When X has full column rank the posterior is
    proper: sigma2 | y ~ IG((n-p-1)/2, (n-p-1) s2 / 2) and
    beta | sigma2, y ~ N(beta-hat, sigma2 (X'X)^-1), with beta-hat the
    OLS estimator and s2 = y'(I - H)y/(n - p - 1) (p.172).
    Keys: estimate."""
    f = _gp.ols_fit(X, y, add_intercept=add_intercept)
    n = len(_gp._flat(y))
    p1 = len(f["beta"])
    df = n - p1
    res = RichResult(payload={"estimate": f["beta"][0],
                              "posterior_mean_beta": f["beta"],
                              "posterior_sd_beta": f["se_beta"],
                              "sigma2_hat": f["sigma2"],
                              "ig_shape": df / 2.0,
                              "ig_scale": df * f["sigma2"] / 2.0,
                              "posterior_mean_sigma2":
                                  (df * f["sigma2"] / 2.0)
                                  / (df / 2.0 - 1.0)
                                  if df > 2 else float("nan"),
                              "method": "reference-prior Bayesian linear regression (MVSML 2022 eq. 6.1-6.2)"})
    return with_describe_pointer(res, "msm042")


def cheatsheet():
    return "msm042: Normal linear regression under the Bayesian paradigm"
