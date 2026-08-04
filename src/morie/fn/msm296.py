# morie.fn -- function file (rootcoder007/morie)

"""Functional regression with environment interaction.

Implements eq. (14.14) p.610 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["fregint", "mvsml_convolutional_nn_eq_14_14"]


def fregint(y, X, X_E, X_EF=None, env=None, lam=0.0, P=None):

    """y = 1_n mu + X_E beta_E + X beta + X_EF beta_EF + e (eq. 14.14),
    which adds to (14.13) the environment-by-reflectance interaction.
    X_EF is the block-diagonal design printed on p.610: record i in
    environment e contributes x_i' in the columns belonging to e and
    zeros elsewhere, so with I environments and L1 functional scores
    it is of order n x (I L1).  Pass env, the environment label of
    each record, to have that matrix built, or pass X_EF directly.
    (14.14) differs from (14.13) by this block alone.

    Written out for all I environments those blocks sum column by
    column to X exactly, so the joint design is rank deficient and
    beta and beta_EF are not separately identified by least squares;
    the book fits (14.14) in BGLR, where the prior on each block
    resolves that.  Building X_EF from env therefore drops the first
    environment block by default, the same reference coding the book
    applies to the environment design itself on p.607, where its code
    reads X_E = model.matrix(~0+Env, data = dat_F)[, -1].
    Keys: coef, mu, beta_E, beta, beta_EF, widths, design, fitted,
    residuals, sse, n_columns, has_interaction, X_EF.
    """

    if X_EF is None and env is not None:
        X_EF = _gp.fda_env_interaction_design(X, env)["X_EF"]
    d = _gp.fda_env_model(y, X, X_E, X_EF=X_EF, lam=lam, P=P)
    d["X_EF"] = X_EF
    res = RichResult(payload=d)

    return with_describe_pointer(res, "msm296")


mvsml_convolutional_nn_eq_14_14 = fregint


def cheatsheet():
    return "msm296: Functional regression with environment interaction"
