# morie.fn -- function file (rootcoder007/morie)
"""Multi-trait model with genotype-by-environment interaction.

Implements eq. (5.6) p.155 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T,
         I_env=None, X=None):
    """Y = (1_IJ (x) I_nT) mu + X beta + Z_L b_1 + Z_EL b_2 + eps
    (eq. 5.6) with b_1 ~ N(0, G (x) Sigma_T) and
    b_2 ~ N(0, Sigma_E (x) G (x) Sigma_2T).  When Sigma_T, Sigma_2T,
    Sigma_E and R are all diagonal this reduces to separate univariate
    GBLUP fits per trait (book p.155). Keys: estimate."""
    f = _gp.gxe_multitrait_model(Y, Z_L, Z_EL, G, Sigma_T, Sigma_E,
                                 Sigma_2T, R_T, I_env, X=X)
    res = RichResult(payload={"estimate": f["mu"][0], "mu": f["mu"],
                              "b_lines": f["b_lines"],
                              "b_gxe": f["b_gxe"],
                              "method": "multi-trait G x E LMM (MVSML 2022 eq. 5.6)"})
    return with_describe_pointer(res, "msm032")


def cheatsheet():
    return "msm032: Multi-trait model with genotype-by-environment interaction"
