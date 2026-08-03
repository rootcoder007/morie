# morie.fn -- function file (rootcoder007/morie)
"""Dual optimization problem.

Implements eq. (9.32) p.349 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_32", "mvsml_svm_dual_objective"]


def mvsml_ridge_lasso_elastic_eq_9_32(alpha, X, y, K=None, fit=False, C=None):
    """maximize L(alpha) = sum_i alpha_i
    - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
    (eq. 9.32).  The problem is cast entirely in inner products of the
    data, never the vectors themselves, which is what lets a kernel
    stand in for a transformation into a higher-dimensional space.
    L is quadratic in alpha with a positive semi-definite Hessian, so
    this is a convex program with a unique solution.  With
    ``fit=True`` the dual is maximized subject to (9.33).
    Keys: estimate."""
    if fit:
        f = _gp.svm_fit_dual(X, y, C=C, K=K)
        res = RichResult(payload={"estimate": f["objective"],
                                  "L": f["objective"],
                                  "alpha": f["alpha"],
                                  "beta": f["beta"],
                                  "beta0": f["beta0"],
                                  "support_vectors":
                                      f["support_vectors"],
                                  "method": "dual problem, fitted (MVSML 2022 eq. 9.32)"})
    else:
        val = _gp.svm_dual_objective(alpha, X, y, K=K)
        res = RichResult(payload={"estimate": val, "L": val,
                                  "method": "dual objective (MVSML 2022 eq. 9.32)"})
    return with_describe_pointer(res, "msm212")


mvsml_svm_dual_objective = mvsml_ridge_lasso_elastic_eq_9_32


def cheatsheet():
    return "msm212: Dual optimization problem"
