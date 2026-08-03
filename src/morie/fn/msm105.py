# morie.fn -- function file (rootcoder007/morie)
"""Ordinal probit GBLUP model.

Implements eq. (7.2) p.214 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_2"]


def mvsml_bayesian_regression_pt2_eq_7_2(y, G, n_iter=1200, burn_in=300, seed=42):
    """p_ic = Phi(gamma_c + b_i) - Phi(gamma_{c-1} + b_i)
    (eq. 7.2) with b | sigma2_g ~ N(0, sigma2_g G): the ordinal
    counterpart of the GBLUP model, fitted by the Gibbs sampler of
    p.214 in which b | . ~ N(b-tilde, (sigma_g^-2 G^-1 + I)^-1) and
    sigma2_g | . ~ chi^-2(nu_g + n, S_g + b'G^-1 b). Keys: estimate."""
    f = _gp.ordinal_probit_gblup_gibbs(y, G, n_iter=n_iter,
                                       burn_in=burn_in, seed=seed)
    probs = _gp.ordinal_probabilities(f["b"], f["gamma"],
                                      link="probit")
    res = RichResult(payload={"estimate": f["b"][0], "b": f["b"],
                              "gamma": f["gamma"],
                              "sigma2_g": f["sigma2_g"],
                              "probabilities": probs,
                              "method": "ordinal probit GBLUP (MVSML 2022 eq. 7.2)"})
    return with_describe_pointer(res, "msm105")


def cheatsheet():
    return "msm105: Ordinal probit GBLUP model"
