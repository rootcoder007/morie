# morie.fn -- function file (rootcoder007/morie)
"""Support vector classifier fitting function.

Implements eq. (9.5) p.341 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_5", "mvsml_svm_decision_function"]


def mvsml_ridge_lasso_elastic_eq_9_5(X, beta0, beta):
    """f(x_i) = beta_0 + x_i' beta (eq. 9.5): a test observation is
    labelled 1 when f-hat is positive and -1 when negative, and large
    magnitudes mean more confidence in the assignment.  For a
    separable training set y_i f(x_i) > 0 holds for every observation.
    Keys: estimate."""
    v = _gp.svm_decision_values(X, beta0, beta)
    res = RichResult(payload={"estimate": v[0], "f": v,
                              "labels": [1 if u > 0 else -1
                                         for u in v],
                              "method": "SVM fitting function (MVSML 2022 eq. 9.5)"})
    return with_describe_pointer(res, "msm173")


mvsml_svm_decision_function = mvsml_ridge_lasso_elastic_eq_9_5


def cheatsheet():
    return "msm173: Support vector classifier fitting function"
