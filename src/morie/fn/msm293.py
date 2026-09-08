# morie.fn -- function file (rootcoder007/morie)

"""Functional regression with environment effects.

Implements eq. (14.13) p.607 of Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).

Note: the generated stub name this module replaces carried a
topic label taken from the wrong chapter, and its body ignored the
cited equation entirely; the name and the implementation below follow
the equation actually printed on that page.
"""

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["fregenv", "mvsml_convolutional_nn_eq_14_13"]


def fregenv(y, X, X_E, lam=0.0, P=None):

    """y = 1_n mu + X_E beta_E + X beta + e (eq. 14.13), the functional
    regression of the chapter extended with the effects of the
    environments.  X carries the L1 functional scores of (14.4) and
    (14.5), X_E is the design matrix of the environments and beta_E
    the environment effects.  In the Bayesian treatment the block
    beta_E is simply given its own prior -- FIXED, BRR, BayesA,
    BayesB, BayesC or BL -- which is why the fit is organized by
    block here.  Keys: coef, mu, beta_E, beta, beta_EF, widths,
    design, fitted, residuals, sse, n_columns, has_interaction.
    """

    res = RichResult(payload=_gp.fda_env_model(y, X, X_E, X_EF=None, lam=lam, P=P))

    return with_describe_pointer(res, "msm293")


mvsml_convolutional_nn_eq_14_13 = fregenv


def cheatsheet():
    return "msm293: Functional regression with environment effects"
