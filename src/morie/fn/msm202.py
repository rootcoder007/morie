# morie.fn -- function file (rootcoder007/morie)
"""Weights as a combination of training vectors.

Implements eq. (9.28) p.348 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_28", "mvsml_svm_beta_from_alpha"]


def mvsml_ridge_lasso_elastic_eq_9_28(alpha, X, y):
    """dL/dbeta = beta - sum_i alpha_i y_i x_i = 0, so
    beta = sum_i alpha_i y_i x_i (eq. 9.28): every beta coefficient
    except the intercept is a linear combination of the training
    vectors, and a vector enters that expansion if and only if
    alpha_i != 0 -- those are the support vectors. Keys: estimate."""
    b = _gp.svm_beta_from_alpha(alpha, X, y)
    sv = [i for i, v in enumerate(_gp._flat(alpha)) if abs(v) > 1e-9]
    res = RichResult(payload={"estimate": b[0], "beta": b,
                              "support_vectors": sv,
                              "method": "weights from multipliers (MVSML 2022 eq. 9.28)"})
    return with_describe_pointer(res, "msm202")


mvsml_svm_beta_from_alpha = mvsml_ridge_lasso_elastic_eq_9_28


def cheatsheet():
    return "msm202: Weights as a combination of training vectors"
