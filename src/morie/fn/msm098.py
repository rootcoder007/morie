# morie.fn -- function file (rootcoder007/morie)
"""Ordinal predictor with environment and genetic effects.

Implements eq. (7.5) p.221 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_5"]


def mvsml_bayesian_regression_pt2_eq_7_5(n, X_E, Z_L, L_g=None):
    """L = X_E beta_E + Z_L g + eps (eq. 7.5): environment effects and
    genetic effects, without the line-by-environment interaction
    (p.220).  With ``L_g`` the Cholesky factor of G the genetic block
    enters as Z_L L_g, the design used in Table 7.6 p.233.
    Keys: estimate."""
    f = _gp.ordinal_latent_predictor(int(n), X_E=X_E, Z_L=Z_L,
                                     L_g=L_g)
    res = RichResult(payload={"estimate": float(f["n_columns"]),
                              "design": f["design"],
                              "widths": f["widths"],
                              "method": "ordinal environment + genetic predictor (MVSML 2022 eq. 7.5)"})
    return with_describe_pointer(res, "msm098")


def cheatsheet():
    return "msm098: Ordinal predictor with environment and genetic effects"
