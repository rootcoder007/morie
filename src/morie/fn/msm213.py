# morie.fn -- function file (rootcoder007/morie)
"""Dual feasibility constraints.

Implements eq. (9.33) p.349 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_33", "mvsml_svm_dual_constraints"]


def mvsml_ridge_lasso_elastic_eq_9_33(alpha, y, C=None):
    """alpha_i >= 0 and sum_i alpha_i y_i = 0 (eq. 9.33).  These
    constraints are affine and convex, which together with the
    positive semi-definite Hessian of (9.32) makes the maximization a
    convex problem.  ``C`` adds the soft-margin upper bound.
    Keys: estimate."""
    f = _gp.svm_dual_constraints_ok(alpha, y, C=C)
    res = RichResult(payload={"estimate": 1.0 if f["feasible"]
                              else 0.0,
                              "nonnegative": f["nonnegative"],
                              "balanced": f["balanced"],
                              "bounded": f["bounded"],
                              "feasible": f["feasible"],
                              "method": "dual constraints (MVSML 2022 eq. 9.33)"})
    return with_describe_pointer(res, "msm213")


mvsml_svm_dual_constraints = mvsml_ridge_lasso_elastic_eq_9_33


def cheatsheet():
    return "msm213: Dual feasibility constraints"
