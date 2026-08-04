# morie.fn -- function file (rootcoder007/morie)

"""Quadratic program under one linear inequality.

Implements eq. (9.15), (9.17), (9.18), (9.19), (9.20), (9.21), (9.22), (9.23), (9.24), (9.25), (9.26) p.346 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["qplincon", "mvsml_ridge_lasso_elastic_eq_9_15"]


def qplincon(a, c):

    """minimize z'z subject to a'z >= c, the shape of both illustrative
    examples of the chapter.  Its Wolfe dual is
    L = z'z - 2 alpha (a'z - c) (eq. 9.17); stationarity (eq. 9.18)
    gives z = alpha a, and substituting back leaves
    L(alpha) = -(a'a) alpha^2 + 2 c alpha (eq. 9.19) maximized at
    alpha = c / (a'a) >= 0 (eq. 9.20).

    Illustrative Example 9.1, "minimize x^2 subject to x >= 1"
    (eqs. 9.15-9.16), is a = [1], c = 1: the dual is
    L(alpha) = -alpha^2 + 2 alpha and x = alpha = 1, as printed.
    Illustrative Example 9.2, "minimize x^2 + y^2 subject to
    x + y >= 2" (eqs. 9.21-9.22, dual 9.23-9.26), is a = [1, 1],
    c = 2: the dual is L(alpha) = -2 alpha^2 + 4 alpha and
    x = y = alpha = 1, also as printed.  The two examples are the same
    problem at different a, so one routine answers both.
    Keys: x, alpha, dual_quadratic, dual_linear, dual_value,
    primal_value, constraint, active.
    """

    res = RichResult(payload=_gp.qp_one_linear_constraint(a, c))

    return with_describe_pointer(res, "msm188")


mvsml_ridge_lasso_elastic_eq_9_15 = qplincon


def cheatsheet():
    return "msm188: Quadratic program under one linear inequality"
