# morie.fn -- function file (rootcoder007/morie)
"""Henderson's mixed model equations.

Implements eq. (2.2) p.36 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_preprocessing_eq_2_2"]


def mvsml_preprocessing_eq_2_2(X, Z, y, Sigma_inv, R_inv=None):
    """The MME of eq. (2.2):
    [X'R^-1X  X'R^-1Z; Z'R^-1X  Z'R^-1Z + Sigma^-1][beta; u] =
    [X'R^-1y; Z'R^-1y]. The beta solution is the BLUE and the u
    solution is the BLUP. Keys: estimate."""
    beta, u = _gp.mme_solve(X, Z, y, Sigma_inv, R_inv)
    res = RichResult(payload={"estimate": beta[0], "blue": beta,
                              "blup": u,
                              "method": "Henderson MME (MVSML 2022 eq. 2.2)"})
    return with_describe_pointer(res, "msm241")


def cheatsheet():
    return "msm241: Henderson's mixed model equations"
