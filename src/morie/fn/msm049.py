# morie.fn -- function file (rootcoder007/morie)
"""Bayesian GBLUP genomic model.

Implements eq. (6.4) pp.176-177 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(y, G, n_iter=2000, burn_in=500, nu=5.0, nu_g=5.0, R2=0.5,
         seed=42):
    """Y = 1_n mu + g + eps with g | sigma2_g ~ N(0, sigma2_g G)
    (eq. 6.4), the induced prior obtained from beta ~ N(0, I
    sigma2_beta) through g = X_1 beta, G = X_1 X_1'/p and
    sigma2_g = p sigma2_beta (p.176).  The book notes on p.177 that
    this is exactly the BRR run on X = L with G = L L', which is how
    it is fitted here; g = L beta. Keys: estimate."""
    f = _gp.bayes_gblup_gibbs(y, G, n_iter=n_iter, burn_in=burn_in,
                              nu=nu, nu_g=nu_g, R2=R2, seed=seed)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "g": f["g"], "sigma2": f["sigma2"],
                              "sigma2_g": f["sigma2_g"],
                              "method": "Bayesian GBLUP (MVSML 2022 eq. 6.4)"})
    return with_describe_pointer(res, "msm049")


def cheatsheet():
    return "msm049: Bayesian GBLUP genomic model"
