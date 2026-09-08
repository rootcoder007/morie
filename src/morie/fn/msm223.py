# morie.fn -- function file (rootcoder007/morie)

"""Karush-Kuhn-Tucker conditions of the support vector classifier.

Implements eq. (9.38), (9.39), (9.40), (9.41), (9.42), (9.43) p.356 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svmkkt", "mvsml_ridge_lasso_elastic_eq_9_38"]


def svmkkt(X, y, beta0, beta, alpha, delta, zeta, T):

    """The Wolfe primal of the soft margin problem,
    L = (1/2)||beta||^2 + T sum_i zeta_i
    - sum_i alpha_i [ y_i(beta_0 + x_i beta) - 1 + zeta_i ]
    - sum_i delta_i zeta_i (eq. 9.38), and the five conditions its
    stationarity produces: beta = sum_i alpha_i y_i x_i (eq. 9.39),
    sum_i alpha_i y_i = 0 (eq. 9.40), alpha_i + delta_i = T
    (eq. 9.41), and the two complementary slackness conditions
    alpha_i [ y_i(beta_0 + x_i beta) - 1 + zeta_i ] = 0 (eq. 9.42)
    and delta_i zeta_i = 0 (eq. 9.43).  Each is returned as a
    residual, zero when the condition holds.

    The printed sign of the delta term in (9.38) on p.356 is
    inconsistent with the book's own (9.41): dL/dzeta_i = T - alpha_i
    - delta_i requires that term to enter with a minus.  It is
    implemented with a minus so that (9.41) holds; the book is not
    silently corrected elsewhere.  Keys: L, stationarity_beta,
    balance, multiplier_sum, complementary_alpha,
    complementary_delta, max_residual, kkt_satisfied.
    """

    res = RichResult(payload=_gp.soft_margin_kkt(X, y, beta0, beta, alpha, delta, zeta, T))

    return with_describe_pointer(res, "msm223")


mvsml_ridge_lasso_elastic_eq_9_38 = svmkkt


def cheatsheet():
    return "msm223: Karush-Kuhn-Tucker conditions of the support vector classifier"
