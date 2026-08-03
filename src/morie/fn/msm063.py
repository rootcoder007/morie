# morie.fn -- function file (rootcoder007/morie)
"""RKHS predictor with genotype-by-environment effects.

Implements eq. (6.7) p.186 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_bayesian_regression_eq_6_7"]


def mvsml_bayesian_regression_eq_6_7(Z_L, G, Z_LE=None, I_env=None, sigma2_g=1.0,
         sigma2_ge=1.0):
    """Y = 1_n mu + X_E beta_E + Z_L g + Z_LE gE + eps (eq. 6.7).
    As for eq. (6.5), the predictor terms enter through their
    covariance matrices, K_L = Z_L G Z_L' and
    K_LE = Z_LE (I (x) G) Z_LE' (p.186); those are computed here.
    Keys: estimate."""
    f = _gp.rkhs_covariances(Z_L, G, Z_LE=Z_LE, I_env=I_env,
                             sigma2_g=sigma2_g, sigma2_ge=sigma2_ge)
    K = f["K_L"]
    res = RichResult(payload={"estimate": K[0][0], "K_L": K,
                              "K_LE": f.get("K_LE"),
                              "method": "RKHS G x E covariances (MVSML 2022 eq. 6.7)"})
    return with_describe_pointer(res, "msm063")


def cheatsheet():
    return "msm063: RKHS predictor with genotype-by-environment effects"
