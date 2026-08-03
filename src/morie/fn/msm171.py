# morie.fn -- function file (rootcoder007/morie)
"""Hyperplane decision rule.

Implements eq. (9.4) p.340 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_4", "mvsml_svm_hyperplane_side"]


def mvsml_ridge_lasso_elastic_eq_9_4(X, beta0, beta):
    """beta_0 + beta_1 x_1 + ... + beta_p x_p > 0 puts a point on one
    side of the hyperplane and < 0 on the other (eq. 9.4), so the
    hyperplane divides p-dimensional space into two halves and the
    sign of the left-hand side identifies the half. Keys: estimate."""
    s = _gp.hyperplane_side(X, beta0, beta)
    res = RichResult(payload={"estimate": float(s[0]), "side": s,
                              "method": "hyperplane decision rule (MVSML 2022 eq. 9.4)"})
    return with_describe_pointer(res, "msm171")


mvsml_svm_hyperplane_side = mvsml_ridge_lasso_elastic_eq_9_4


def cheatsheet():
    return "msm171: Hyperplane decision rule"
