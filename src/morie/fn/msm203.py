# morie.fn -- function file (rootcoder007/morie)
"""Balance condition on the multipliers.

Implements eq. (9.29) p.348 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_29", "mvsml_svm_balance_condition"]


def mvsml_ridge_lasso_elastic_eq_9_29(alpha, y):
    """dL/dbeta_0 = -sum_i alpha_i y_i = 0, so sum_i alpha_i y_i = 0
    (eq. 9.29): the stationarity condition in the intercept, which
    becomes one of the two dual constraints. Keys: estimate."""
    a = _gp._flat(alpha)
    ys = _gp._flat(y)
    s = sum(ai * yi for ai, yi in zip(a, ys))
    res = RichResult(payload={"estimate": s, "balance": s,
                              "satisfied": abs(s) < 1e-6,
                              "method": "intercept stationarity (MVSML 2022 eq. 9.29)"})
    return with_describe_pointer(res, "msm203")


mvsml_svm_balance_condition = mvsml_ridge_lasso_elastic_eq_9_29


def cheatsheet():
    return "msm203: Balance condition on the multipliers"
