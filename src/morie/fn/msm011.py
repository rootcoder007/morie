# morie.fn -- function file (rootcoder007/morie)
"""Marginal likelihood of the linear mixed model.

Implements eq. (5.2) p.142 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_2"]


def mvsml_linear_mixed_models_eq_5_2(X, Z, y, D, R=None, beta=None, restricted=False):
    """L(beta, D, R; y) = |V|^(-1/2)(2 pi)^(-n/2)
    exp(-1/2 (y - X beta)' V^-1 (y - X beta)) with V = Z'DZ + R
    (eq. 5.2); the log is returned.  With ``restricted=True`` the
    REML objective of p.146 is returned instead, which adds the
    -1/2 log|X'V^-1X| term that removes the ML bias.
    Keys: estimate."""
    if restricted:
        val, bhat = _gp.reml_loglik(X, Z, y, D, R)
    else:
        val, bhat = _gp.lmm_loglik(X, Z, y, D, beta=beta, R=R)
    res = RichResult(payload={"estimate": val, "loglik": val,
                              "beta": bhat, "restricted": restricted,
                              "method": "LMM marginal likelihood (MVSML 2022 eq. 5.2)"})
    return with_describe_pointer(res, "msm011")


def cheatsheet():
    return "msm011: Marginal likelihood of the linear mixed model"
