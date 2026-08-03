# morie.fn -- function file (rootcoder007/morie)
"""Linear multiple regression fitted by OLS.

Implements eq. (3.1) p.71 with the OLS solution pp.72-73 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_elements_lin_reg_eq_3_1"]


def mvsml_elements_lin_reg_eq_3_1(X, y, add_intercept=True):
    """Y = beta_0 + sum_j X_j beta_j + eps (eq. 3.1) fitted by least
    squares: beta = (X'X)^-1X'y, Var(beta) = sigma2 (X'X)^-1 and
    sigma2 = RSS/(n - p - 1) (pp.72-73). Keys: estimate."""
    f = _gp.ols_fit(X, y, add_intercept=add_intercept)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"], "se": f["se_beta"],
                              "sigma2": f["sigma2"],
                              "fitted": f["fitted"],
                              "residuals": f["residuals"],
                              "method": "OLS linear multiple regression (MVSML 2022 eq. 3.1)"})
    return with_describe_pointer(res, "msm333")


def cheatsheet():
    return "msm333: Linear multiple regression fitted by OLS"
