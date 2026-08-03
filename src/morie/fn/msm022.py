# morie.fn -- function file (rootcoder007/morie)
"""GBLUP model for genomic prediction.

Implements eq. (5.3) p.148 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g, sigma2_e=1.0):
    """Y = 1_n mu + Z_L b + eps with b ~ N_J(0, sigma2_g G) and
    R = sigma2 I_n (eq. 5.3): the GBLUP model.  Z_L is the incidence
    matrix of lines and G the genomic relationship matrix; the BLUP of
    b holds the genomic estimated breeding values. Keys: estimate."""
    f = _gp.gblup_model(y, Z_L, G, sigma2_g, sigma2_e)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "gebv": f["b"],
                              "method": "GBLUP model (MVSML 2022 eq. 5.3)"})
    return with_describe_pointer(res, "msm022")


def cheatsheet():
    return "msm022: GBLUP model for genomic prediction"
