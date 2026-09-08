# morie.fn -- function file (rootcoder007/morie)

"""Wolfe dual of a constrained program.

Implements eq. (9.9), (9.12), (9.13), (9.14) p.346 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["wolfedual", "mvsml_ridge_lasso_elastic_eq_9_9"]


def wolfedual(f, grad_f, h=None, grad_h=None, g=None, grad_g=None, lam=None,
              alpha=None):

    """The Wolfe dual of "minimize f(x) subject to h_i(x) = 0 and
    g_i(x) >= 0" (eq. 9.9 with its constraints 9.10 and 9.11) is
    "maximize L = f - sum_i lambda_i h_i - sum_i alpha_i g_i"
    (eq. 9.12) subject to the stationarity condition
    grad f - sum_i lambda_i grad h_i - sum_i alpha_i grad g_i = 0
    (eq. 9.13) and alpha_i >= 0 (eq. 9.14).  The search moves from an
    n-dimensional space to an (n + m + p)-dimensional one.  The book
    warns under (9.14) that the sign of the inequality term is
    crucial; its own worked examples supply the constraint in the >=
    form and subtract it, which is the convention used here.
    Keys: L, stationarity, max_stationarity, alpha_nonnegative,
    n_equality, n_inequality.
    """

    res = RichResult(payload=_gp.wolfe_dual(f, grad_f, h=h, grad_h=grad_h, g=g,
                     grad_g=grad_g, lam=lam, alpha=alpha))

    return with_describe_pointer(res, "msm184")


mvsml_ridge_lasso_elastic_eq_9_9 = wolfedual


def cheatsheet():
    return "msm184: Wolfe dual of a constrained program"
