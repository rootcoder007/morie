# morie.fn -- function file (rootcoder007/morie)

"""Roughness penalty matrix.

Implements eq. (14.11) p.601 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["penmat", "mvsml_convolutional_nn_eq_14_11"]


def penmat(t, L1, p=2, kind='fourier', period=None, beta=None):

    """J_beta = int_0^T [ d^p beta(t) / dt^p ]^2 dt (eq. 14.11), the
    penalty based on the integrated squared pth order derivative.
    Under the basis expansion (14.2) the book writes it as the
    quadratic form J_beta = beta' P beta with
    P_ij = int_0^T phi_i^(p)(t) phi_j^(p)(t) dt, i, j = 1, ..., L1.
    The chapter says the values of p typically chosen are 1 and 2.
    The integrals are evaluated by the trapezoid rule on the grid t;
    the basis derivatives are analytic.  Keys: P, order, L1, and J
    when beta is supplied.
    """

    res = RichResult(payload=_gp.fda_penalty_matrix(t, L1, p=p, kind=kind, period=period,
                             beta=beta))

    return with_describe_pointer(res, "msm278")


mvsml_convolutional_nn_eq_14_11 = penmat


def cheatsheet():
    return "msm278: Roughness penalty matrix"
