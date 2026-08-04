# morie.fn -- function file (rootcoder007/morie)

"""Functional linear model with scalar response.

Implements eq. (14.1) p.579 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["flmint", "mvsml_convolutional_nn_eq_14_1"]


def flmint(t, x_values, beta_values, mu=0.0):

    """Y = mu + int_0^T x(t) beta(t) dt + E (eq. 14.1).  Functional
    regression replaces the linear predictor of an ordinary
    regression by the integral of the product of a centered covariate
    curve x(t) and a coefficient function beta(t).  The integral is
    taken by the trapezoid rule on the observation grid, the same
    quadrature the chapter uses for its inner products.  Recovering
    beta(t) itself from finitely many observations is ill posed --
    infinitely many functions give the same predictions -- which is
    why the chapter turns to basis expansion (eq. 14.2).
    Keys: integral, fitted, mu, n_points.
    """

    res = RichResult(payload=_gp.fda_integral(t, x_values, beta_values, mu=mu))

    return with_describe_pointer(res, "msm261")


mvsml_convolutional_nn_eq_14_1 = flmint


def cheatsheet():
    return "msm261: Functional linear model with scalar response"
