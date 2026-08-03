# morie.fn -- function file (rootcoder007/morie)
"""Genomic multi-trait linear model.

Implements eq. (6.8) p.191 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_8"]


def mvsml_bayesian_regression_eq_6_8(Y, Z1, G, X=None, n_iter=1200, burn_in=300, seed=42):
    """Y_j = mu + B' x_j + g_j + eps_j (eq. 6.8): a univariate genomic
    structure per trait with correlated residuals and correlated
    genotype effects, eps_j ~ N(0, R) and
    g ~ N(0, G (x) Sigma_T).  Inverse-Wishart priors are placed on
    Sigma_T and R, a flat prior on the intercepts. Keys: estimate."""
    f = _gp.multitrait_bayes_gibbs(Y, Z1, G, X=X, n_iter=n_iter,
                                   burn_in=burn_in, seed=seed)
    res = RichResult(payload={"estimate": f["mu"][0], "mu": f["mu"],
                              "b1": f["b1"], "Sigma_T": f["Sigma_T"],
                              "R": f["R"],
                              "method": "genomic multi-trait model (MVSML 2022 eq. 6.8)"})
    return with_describe_pointer(res, "msm069")


def cheatsheet():
    return "msm069: Genomic multi-trait linear model"
