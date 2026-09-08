# morie.fn -- function file (rootcoder007/morie)
"""Karush-Kuhn-Tucker complementary slackness.

Implements eq. (9.30) p.348 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 9 is Support Vector Machines and Support Vector Regression,
and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_30", "mvsml_svm_kkt_conditions"]


def mvsml_ridge_lasso_elastic_eq_9_30(alpha, X, y, beta0, beta, tol=1e-6):
    """alpha_i [y_i(beta_0 + x_i'beta) - 1] = 0 for every i
    (eq. 9.30), the KKT complementary-slackness condition.  It forces
    the dichotomy of p.348: if alpha_i > 0 then
    y_i(beta_0 + x_i'beta) = 1 and x_i sits on the margin; if
    y_i(beta_0 + x_i'beta) > 1 then x_i is off the margin and
    alpha_i = 0.  So support vectors lie exactly on
    y_i(beta_0 + x_i'beta) = 1. Keys: estimate."""
    a = _gp._flat(alpha)
    ys = _gp._flat(y)
    f = _gp.svm_decision_values(X, beta0, beta)
    slack = [ys[i] * f[i] - 1.0 for i in range(len(a))]
    prod = [a[i] * slack[i] for i in range(len(a))]
    on_margin = [i for i in range(len(a)) if abs(slack[i]) < tol]
    res = RichResult(payload={"estimate": max(abs(v) for v in prod),
                              "complementary_products": prod,
                              "margin_slack": slack,
                              "on_margin": on_margin,
                              "satisfied": max(abs(v) for v in prod)
                              < tol,
                              "method": "KKT conditions (MVSML 2022 eq. 9.30)"})
    return with_describe_pointer(res, "msm204")


mvsml_svm_kkt_conditions = mvsml_ridge_lasso_elastic_eq_9_30


def cheatsheet():
    return "msm204: Karush-Kuhn-Tucker complementary slackness"
