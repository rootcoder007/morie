# morie.fn -- function file (rootcoder007/morie)
"""Ordinal latent predictor with environment, markers and interaction.

Implements eq. (7.3) p.219 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def mvsml_bayesian_regression_pt2_eq_7_3(n, X_E=None, X=None, X_EM=None):
    """L = X_E beta_E + X beta + X_EM beta_EM + eps (eq. 7.3): the
    ordinal latent variable with a flat prior on the environment
    effects and a BRR/BayesA/BayesB/BayesC/BL prior on the marker and
    marker-by-environment effects (p.219). Keys: estimate."""
    f = _gp.ordinal_latent_predictor(int(n), X_E=X_E, X=X,
                                     X_EM=X_EM)
    res = RichResult(payload={"estimate": float(f["n_columns"]),
                              "design": f["design"],
                              "widths": f["widths"],
                              "method": "ordinal latent predictor (MVSML 2022 eq. 7.3)"})
    return with_describe_pointer(res, "msm097")


def cheatsheet():
    return "msm097: Ordinal latent predictor with environment, markers and interaction"
