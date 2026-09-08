# morie.fn -- function file (rootcoder007/morie)
"""Multivariate ridge form of the multi-trait model.

Implements eq. (6.10) p.194 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["multitrait_ridge_form"]


def multitrait_ridge_form(Z1, G):
    """Y = 1_J mu' + X B + X_1 B_1 + E (eq. 6.10), the multivariate
    ridge regression form of eq. (6.9): X_1 = Z_1 L_G with
    G = L_G L_G' the Cholesky factorization and
    B_1 = L_G^-1 b_1 ~ MN(0, I_J, Sigma_T), so the RKHS predictor is
    replaced by a BRR one on X_1. Keys: estimate."""
    f = _gp.multitrait_ridge_form(Z1, G)
    res = RichResult(payload={"estimate": f["X1"][0][0],
                              "X1": f["X1"], "L_G": f["L_G"],
                              "method": "multivariate ridge form (MVSML 2022 eq. 6.10)"})
    return with_describe_pointer(res, "msm072")


def cheatsheet():
    return "msm072: Multivariate ridge form of the multi-trait model"
