# morie.fn -- function file (rootcoder007/morie)

"""Basis expansion of the coefficient function.

Implements eq. (14.2) p.579 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["basexp", "mvsml_convolutional_nn_eq_14_2"]


def basexp(t, beta_coef, kind='fourier', period=None):

    """beta(t) = sum_{l=1}^{L1} beta_l phi_l(t) (eq. 14.2), where the
    phi_l are the first L1 elements of a basis for a function space
    (Fourier, B-spline, polynomial) and the beta_l are constants.
    This is the device that makes (14.1) estimable: it replaces an
    infinite-dimensional unknown function by L1 scalars, after which
    (14.1) collapses to the ordinary linear model (14.3).
    Keys: beta_t, t, n_basis.
    """

    coefs = list(beta_coef)
    vals = _gp.fda_beta_function(t, coefs, len(coefs), kind=kind)
    res = RichResult(payload={"beta_t": vals, "t": list(t),
                              "n_basis": len(coefs)})

    return with_describe_pointer(res, "msm262")


mvsml_convolutional_nn_eq_14_2 = basexp


def cheatsheet():
    return "msm262: Basis expansion of the coefficient function"
