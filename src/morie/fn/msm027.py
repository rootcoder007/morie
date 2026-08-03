# morie.fn -- function file (rootcoder007/morie)
"""General univariate linear mixed model.

Implements eq. (5.1) p.142 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_1"]


def mvsml_linear_mixed_models_eq_5_1(X, Z, y, D, R=None):
    """Y = X beta + Z b + eps (eq. 5.1) with b ~ N(0, D),
    eps ~ N(0, R) and Cov(eps, b) = 0, so E(Y) = X beta and
    Var(Y) = Z D Z' + R.  Returns the GLS/BLUE of beta, the BLUP of b
    and the marginal variance. Keys: estimate."""
    beta, b = _gp.blue_blup_via_v(X, Z, y, D, R)
    V = _gp.lmm_marginal_v(Z, D, R)
    ll, _ = _gp.lmm_loglik(X, Z, y, D, beta=beta, R=R)
    res = RichResult(payload={"estimate": beta[0], "beta": beta,
                              "blup": b, "V": V, "loglik": ll,
                              "method": "general linear mixed model (MVSML 2022 eq. 5.1)"})
    return with_describe_pointer(res, "msm027")


def cheatsheet():
    return "msm027: General univariate linear mixed model"
