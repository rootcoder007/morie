# morie.fn -- function file (rootcoder007/morie)

"""Penalized sum of squared errors.

Implements eq. (14.10) p.599 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["pensse", "mvsml_convolutional_nn_eq_14_10"]


def pensse(y, X, beta, lam, P, mu=0.0):

    """SSE_lambda(beta) = sum_i ( y_i - mu - sum_l x_il beta_l )^2
    + lambda J_beta (eq. 14.10), where J_beta is the roughness
    penalty of (14.11) and lambda sets the compromise between fit to
    the data (first term) and smoothness of beta() (second term).  At
    lambda = 0 the problem reduces to least squares, and as lambda
    grows the roughness is penalized so heavily that beta(t) is
    driven towards a constant.  Keys: sse, penalty, lambda,
    objective, fitted, residuals.
    """

    res = RichResult(payload=_gp.fda_penalized_sse(y, X, beta, lam, P, mu=mu))

    return with_describe_pointer(res, "msm277")


mvsml_convolutional_nn_eq_14_10 = pensse


def cheatsheet():
    return "msm277: Penalized sum of squared errors"
