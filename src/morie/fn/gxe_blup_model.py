# morie.fn -- function file (rootcoder007/morie)
"""Genotype-by-environment BLUP model.

Implements eq. (5.4) p.150 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["gxe_blup_model"]


def gxe_blup_model(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e=1.0):
    """Y = 1_n mu + X_E beta_E + Z_L b_1 + Z_EL b_2 + eps (eq. 5.4):
    the G x E BLUP model, with b_1 ~ N_J(0, sigma2_g G) the genotypic
    effects and b_2 ~ N(0, Sigma_E (x) G) the genotype-by-environment
    interaction, Sigma_E the genetic covariance between environments.
    Keys: estimate."""
    f = _gp.gxe_blup_model(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E,
                           sigma2_e)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"],
                              "b_lines": f["b_lines"],
                              "b_gxe": f["b_gxe"],
                              "method": "G x E BLUP model (MVSML 2022 eq. 5.4)"})
    return with_describe_pointer(res, "msm018")


def cheatsheet():
    return "msm018: Genotype-by-environment BLUP model"


# compact alias per ledger/NAMING.md
gxeblupmodel = gxe_blup_model
