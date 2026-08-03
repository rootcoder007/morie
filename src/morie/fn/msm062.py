# morie.fn -- function file (rootcoder007/morie)
"""Extended predictor with environment and interaction terms.

Implements eq. (6.6) p.186 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_6"]


def mvsml_bayesian_regression_eq_6_6(n, X_E=None, X=None, X_EM=None):
    """y = 1_n mu + X_E beta_E + X beta + X_EM beta_EM + eps
    (eq. 6.6): environments and environment-by-marker interactions
    added to the predictor of eq. (6.1).  Each block can carry its own
    prior (FIXED, BRR, BayesA/B/C, BL) in BGLR, so the assembled
    design is returned together with the block widths.
    Keys: estimate."""
    f = _gp.extended_predictor(int(n), X_E=X_E, X=X, X_EM=X_EM)
    res = RichResult(payload={"estimate": float(f["n_columns"]),
                              "design": f["design"],
                              "widths": f["widths"],
                              "method": "extended Bayesian predictor (MVSML 2022 eq. 6.6)"})
    return with_describe_pointer(res, "msm062")


def cheatsheet():
    return "msm062: Extended predictor with environment and interaction terms"
