# morie.fn -- function file (rootcoder007/morie)

"""Maximum margin classifier.

Implements eq. (9.6), (9.7), (9.8) p.344 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["hardsvm", "mvsml_ridge_lasso_elastic_eq_9_6"]


def hardsvm(X, y):

    """maximize M subject to sum_j beta_j^2 = 1 and
    y_i(beta_0 + x_i beta) >= M (eq. 9.6).  Because the margin is
    M = 1 / ||beta|| once the scale is fixed, (9.6) is equivalent to
    minimizing (1/2)||beta||^2 (eq. 9.7) subject to
    y_i(beta_0 + x_i beta) >= 1 (eq. 9.8), and the whole street is
    2M = 2 / ||beta||.  Only the observations lying on the margin --
    the support vectors -- determine the solution.
    Keys: beta, beta0, margin, street_width, objective, norm_beta,
    functional_margin, min_functional_margin, constraint_ok, alpha,
    support_vectors.
    """

    res = RichResult(payload=_gp.max_margin_classifier(X, y))

    return with_describe_pointer(res, "msm175")


mvsml_ridge_lasso_elastic_eq_9_6 = hardsvm


def cheatsheet():
    return "msm175: Maximum margin classifier"
