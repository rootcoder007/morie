# morie.fn -- function file (rootcoder007/morie)
"""SNP-BLUP mixed model equation.

Implements eq. (2.4) p.53 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["snp_blup_gebv"]


def snp_blup_gebv(X, y, M, sigma2_m, sigma2_e=1.0):
    """SNP-BLUP: eq. (2.2) with Z = M (the scaled marker matrix) and
    Sigma = sigma2_M I (eq. 2.4). Here u holds marker effects and the
    GEBV is M u-hat; the book notes GBLUP and SNP-BLUP give the same
    breeding values. Keys: estimate."""
    beta, u, gebv = _gp.snp_blup_gebv(X, y, M, sigma2_m, sigma2_e)
    res = RichResult(payload={"estimate": gebv[0], "gebv": gebv,
                              "marker_effects": u, "beta": beta,
                              "method": "SNP-BLUP MME (MVSML 2022 eq. 2.4)"})
    return with_describe_pointer(res, "msm243")


def cheatsheet():
    return "msm243: SNP-BLUP mixed model equation"
