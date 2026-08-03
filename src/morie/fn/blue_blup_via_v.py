# morie.fn -- function file (rootcoder007/morie)
"""Linear mixed model.

Implements eq. (2.1) p.36 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["blue_blup_via_v"]


def blue_blup_via_v(X, Z, y, Sigma, R=None):
    """Y = X beta + Z u + eps with u ~ N(0, Sigma), eps ~ N(0, R)
    (eq. 2.1). V = Z Sigma Z' + R; the BLUE is
    beta = (X'V^-1X)^-1X'V^-1y and the BLUP is u = Sigma Z'V^-1(y - X
    beta). Keys: estimate."""
    beta, u = _gp.blue_blup_via_v(X, Z, y, Sigma, R)
    res = RichResult(payload={"estimate": beta[0], "blue": beta,
                              "blup": u,
                              "method": "linear mixed model, V-based solution (MVSML 2022 eq. 2.1)"})
    return with_describe_pointer(res, "msm240")


def cheatsheet():
    return "msm240: Linear mixed model"
