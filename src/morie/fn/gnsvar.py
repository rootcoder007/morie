# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the GNS regression coefficients given rho, lambda and sigma^2."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from ._qpcore import inverse, ssum


def gnsvar(X, W, rho, lam, sigma2):
    """GNS coefficient covariance ``sigma^2 (X_*' X_*)^{-1}``, ``X_* = (I - lambda W) X``; ``statistic`` its trace.

    ``X`` is the full design (``[X, W X]`` for the Durbin part); ``rho``
    enters only through the fitted ``sigma^2`` (the covariance of ``beta``
    conditional on the spatial parameters).

    Examples
    --------
    >>> r = gnsvar([[1, 0.0], [1, 1.0], [1, 2.0]], [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], 0.2, 0.0, 1.0)
    >>> [round(v, 6) for v in r.extra["cov"][1]]
    [-0.5, 0.5]
    """
    Xm = np.asarray(X, dtype=float).tolist()
    Wm = np.asarray(W, dtype=float).tolist()
    n, p = len(Xm), len(Xm[0])
    WX = [[ssum(Wm[i][m] * Xm[m][k] for m in range(n)) for k in range(p)] for i in range(n)]
    Xs = [[Xm[i][k] - float(lam) * WX[i][k] for k in range(p)] for i in range(n)]
    V = [
        [v * float(sigma2) for v in row]
        for row in inverse([[ssum(r[a] * r[b] for r in Xs) for b in range(p)] for a in range(p)])
    ]
    return SpatialResult(name="gnsvar", statistic=ssum(V[k][k] for k in range(p)), p_value=None, extra={"cov": V})


gnsvar_fn = gnsvar


def cheatsheet() -> str:
    return "gnsvar({}) -> GNS variance-covariance matrix."
