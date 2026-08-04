# morie.fn -- function file (rootcoder007/morie)

"""Hyperplane definition and side.

Implements eq. (9.1), (9.2), (9.3) p.339 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["hyperpl", "mvsml_ridge_lasso_elastic_eq_9_1"]


def hyperpl(X, beta0, beta):

    """beta_0 + beta_1 x_1 + beta_2 x_2 + beta_3 x_3 = 0 defines a
    hyperplane (eq. 9.1, the p = 3 case), and its p-dimensional
    extension beta_0 + beta_1 x_1 + ... + beta_p x_p = 0 (eq. 9.2)
    defines a (p-1)-dimensional flat subspace.  Points whose left-hand
    side is < 0 satisfy (9.3) and lie on one side of it; those with it
    > 0 satisfy (9.4) and lie on the other, so the sign of the
    left-hand side alone says which half of the space a point is in.
    |f(x)| / ||beta|| is the Euclidean distance to the plane.
    Keys: value, side, below, above, on_plane, distance, norm_beta.
    """

    res = RichResult(payload=_gp.hyperplane_value(X, beta0, beta))

    return with_describe_pointer(res, "msm161")


mvsml_ridge_lasso_elastic_eq_9_1 = hyperpl


def cheatsheet():
    return "msm161: Hyperplane definition and side"
