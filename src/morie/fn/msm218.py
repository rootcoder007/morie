# morie.fn -- function file (rootcoder007/morie)

"""Support vector classifier (soft margin).

Implements eq. (9.34), (9.35), (9.36), (9.37) p.354 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["softsvm", "mvsml_ridge_lasso_elastic_eq_9_34"]


def softsvm(X, y, T):

    """maximize M (eq. 9.34) subject to sum_j beta_j^2 = 1 (eq. 9.35),
    y_i(beta_0 + sum_j beta_j x_ij) >= M(1 - zeta_i) (eq. 9.36) and
    zeta_i >= 0 with sum_i zeta_i <= T (eq. 9.37).  The slack zeta_i
    says where observation i sits: zeta_i = 0 is the correct side of
    the margin, zeta_i > 0 has broken the margin, and zeta_i > 1 is on
    the wrong side of the hyperplane altogether.  A larger budget T
    widens the margin and admits more support vectors.

    The book writes T both for the slack budget of (9.37) and for the
    box bound on the multipliers in (9.45).  Those are different
    parameters in the standard formulation and only (9.45) is
    directly solvable, so T here is the box bound and the realized
    sum of slacks is reported as slack_sum for comparison against a
    budget.  Keys: beta, beta0, margin, norm_beta, zeta, slack_sum,
    n_violating, n_misclassified, alpha, support_vectors,
    objective.
    """

    res = RichResult(payload=_gp.soft_margin_classifier(X, y, T))

    return with_describe_pointer(res, "msm218")


mvsml_ridge_lasso_elastic_eq_9_34 = softsvm


def cheatsheet():
    return "msm218: Support vector classifier (soft margin)"
