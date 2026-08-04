# morie.fn -- function file (rootcoder007/morie)

"""Wolfe primal of the maximum margin problem.

Implements eq. (9.27) p.348 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svmlagr", "mvsml_ridge_lasso_elastic_eq_9_27"]


def svmlagr(X, y, beta0, beta, alpha):

    """L(beta, beta_0, alpha) = (1/2)||beta||^2
    - sum_i alpha_i [ y_i(beta_0 + x_i beta) - 1 ] (eq. 9.27), where
    the alpha_i are nonnegative Lagrange multipliers.  Setting the
    derivatives with respect to beta and beta_0 to zero gives
    beta = sum_i alpha_i y_i x_i (eq. 9.28) and sum_i alpha_i y_i = 0
    (eq. 9.29), returned here as grad_beta and grad_beta0, both zero
    at the optimum.  Complementary slackness (eq. 9.30) makes alpha_i
    zero for every observation strictly off the margin, which is why
    the solution depends only on the support vectors.
    Keys: L, quadratic_term, slack, grad_beta, grad_beta0.
    """

    res = RichResult(payload=_gp.svm_lagrangian(X, y, beta0, beta, alpha))

    return with_describe_pointer(res, "msm201")


mvsml_ridge_lasso_elastic_eq_9_27 = svmlagr


def cheatsheet():
    return "msm201: Wolfe primal of the maximum margin problem"
