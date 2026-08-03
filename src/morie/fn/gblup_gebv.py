# morie.fn -- function file (rootcoder007/morie)
"""GBLUP mixed model equation.

Implements eq. (2.3) p.53 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["gblup_gebv"]


def gblup_gebv(X, y, G, sigma2_g, sigma2_e=1.0):
    """GBLUP: eq. (2.2) with Z = 1 (an incidence matrix of lines) and
    Sigma = sigma2_g G, so the second block carries sigma2_g^-2 G^-1
    (eq. 2.3). The u solution is the vector of GEBVs.
    Keys: estimate."""
    beta, u = _gp.gblup_gebv(X, y, G, sigma2_g, sigma2_e)
    res = RichResult(payload={"estimate": u[0], "gebv": u,
                              "beta": beta,
                              "method": "GBLUP MME (MVSML 2022 eq. 2.3)"})
    return with_describe_pointer(res, "msm242")


def cheatsheet():
    return "msm242: GBLUP mixed model equation"
