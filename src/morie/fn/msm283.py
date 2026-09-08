# morie.fn -- function file (rootcoder007/morie)

"""Penalized functional regression fit.

Implements eq. (14.12) p.601 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["penfreg", "mvsml_convolutional_nn_eq_14_12"]


def penfreg(y, X, P, lam, mu=None):

    """With the spectral decomposition P = Gamma D Gamma' of the
    penalty matrix, X* = X Gamma and beta* = Gamma' beta, the
    penalized criterion (14.10) becomes
    SSE_lambda(beta*) = ||y - 1_n mu - X* beta*||^2
    + lambda beta*' D beta* (eq. 14.12), a ridge regression on
    rotated columns with a diagonal penalty.  Its minimizer is
    beta* = (X*'X* + lambda D)^-1 X*'(y - 1_n mu) and the original
    coefficients are recovered as beta = Gamma beta*, after which
    beta-hat(t) = sum_l beta-hat_l phi_l(t).  When P is rank
    deficient the zero eigenvalues contribute nothing, which is the
    reduction to lambda beta_1*' D_1 beta_1* the book notes.
    Keys: beta, beta_star, Gamma, eigenvalues, X_star, mu, fitted,
    residuals, sse, penalty, objective, rank.
    """

    res = RichResult(payload=_gp.fda_penalized_fit(y, X, P, lam, mu=mu))

    return with_describe_pointer(res, "msm283")


mvsml_convolutional_nn_eq_14_12 = penfreg


def cheatsheet():
    return "msm283: Penalized functional regression fit"
