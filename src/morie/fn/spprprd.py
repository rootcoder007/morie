# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit predicted probabilities."""

import math

from ._qpcore import inverse, ssum
from .spprmf import _rows


def spprprd(coef, X, W, rho=0.2, method="marginal"):
    r"""Predicted probabilities P(y_i = 1) of the SAR probit model.

    With S = I - rho W, y* = S^{-1}(X beta + e) has mean eta = S^{-1} X
    beta and marginal variances s_i^2 = ((S'S)^{-1})_ii, so the exact
    marginal probability is Phi(eta_i / s_i) (method="marginal";
    LeSage and Pace 2009, sec. 10.1.4); method="lesage_pace" gives the
    unscaled Phi(eta_i) used for the impacts. Returns the list of
    probabilities.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> [round(v, 10) for v in spprprd([-0.2, 0.9], X, W, rho=0.4)][:2]
    [0.85554162, 0.2356029578]
    """
    b = [float(v) for v in coef]
    Xm, Wm = _rows(X), _rows(W)
    n = len(Xm)
    Si = inverse([[(1.0 if i == j else 0.0) - float(rho) * Wm[i][j] for j in range(n)] for i in range(n)])
    xb = [ssum(r[k] * b[k] for k in range(len(b))) for r in Xm]
    eta = [ssum(Si[i][j] * xb[j] for j in range(n)) for i in range(n)]
    if method == "marginal":
        sd = [math.sqrt(ssum(Si[i][m] * Si[i][m] for m in range(n))) for i in range(n)]
    elif method == "lesage_pace":
        sd = [1.0] * n
    else:
        raise ValueError("method must be 'marginal' or 'lesage_pace'")
    return [0.5 * math.erfc(-e / s / math.sqrt(2.0)) for e, s in zip(eta, sd)]


spprprd_fn = spprprd


def cheatsheet() -> str:
    return "spprprd(coef, X, W, rho) -> SAR probit probabilities Phi(eta_i / s_i), eta = S^-1 X beta."
