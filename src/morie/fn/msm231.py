# morie.fn -- function file (rootcoder007/morie)

"""Wolfe dual of the support vector classifier.

Implements eq. (9.44), (9.45) p.357 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svmsdual", "mvsml_ridge_lasso_elastic_eq_9_44"]


def svmsdual(X, y, T, K=None):

    """maximize L(alpha) = sum_i alpha_i
    - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
    (eq. 9.44) subject to 0 <= alpha_i <= T and sum_i alpha_i y_i = 0
    (eq. 9.45).  It differs from the hard margin dual (9.32)-(9.33)
    only by the upper bound T on the multipliers, which is what the
    slack variables buy.  The objective is concave and infinitely
    differentiable, so this is a convex quadratic program.  Keys:
    alpha, beta, beta0, objective, support_vectors, balance, bounded,
    at_bound.
    """

    res = RichResult(payload=_gp.svm_soft_dual(X, y, T, K=K))

    return with_describe_pointer(res, "msm231")


mvsml_ridge_lasso_elastic_eq_9_44 = svmsdual


def cheatsheet():
    return "msm231: Wolfe dual of the support vector classifier"
