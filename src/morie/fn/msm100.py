# morie.fn -- function file (rootcoder007/morie)
"""Ordinal GBLUP regression model.

Implements eq. (7.4) p.220 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_4"]


def mvsml_bayesian_regression_pt2_eq_7_4(y, G, n_iter=800, burn_in=200, seed=42):
    """L = Z_L g + eps with g ~ N(0, sigma2_g G) (eq. 7.4): the
    ordinal GBLUP regression model, in which the genomic relationship
    matrix enters as an RKHS kernel rather than through a marker
    design matrix (p.220). Keys: estimate."""
    f = _gp.ordinal_probit_gblup_gibbs(y, G, n_iter=n_iter,
                                       burn_in=burn_in, seed=seed)
    res = RichResult(payload={"estimate": f["b"][0], "b": f["b"],
                              "gamma": f["gamma"],
                              "sigma2_g": f["sigma2_g"],
                              "method": "ordinal GBLUP (MVSML 2022 eq. 7.4)"})
    return with_describe_pointer(res, "msm100")


def cheatsheet():
    return "msm100: Ordinal GBLUP regression model"
