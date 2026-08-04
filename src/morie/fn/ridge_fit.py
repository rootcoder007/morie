# morie.fn -- function file (rootcoder007/morie)
"""Ridge regression, penalized least squares.

Implements sec. 3.6.1 p.81 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ridge_fit"]


def ridge_fit(X, y, lam=1.0, add_intercept=True):
    """PRSS_lambda(beta) = RSS(beta) + lambda beta'D beta with
    D = diag(0, 1, ..., 1), giving
    beta^R(lambda) = (X'X + lambda D)^-1 X'y (p.81). The intercept is
    never penalized; lambda = 0 recovers OLS. Keys: estimate."""
    f = _gp.ridge_fit(X, y, lam, add_intercept=add_intercept)
    res = RichResult(payload={"estimate": f["beta"][0],
                              "beta": f["beta"], "prss": f["prss"],
                              "rss": f["rss"], "penalty": f["penalty"],
                              "method": "ridge regression (MVSML 2022 sec. 3.6.1)"})
    return with_describe_pointer(res, "msm258")


def cheatsheet():
    return "msm258: Ridge regression, penalized least squares"


# compact alias per ledger/NAMING.md
ridgefit = ridge_fit
