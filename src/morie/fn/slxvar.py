# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the SLX (spatial lag of X) OLS estimator."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from ._qpcore import inverse, ssum


def slxvar(X, W, sigma2: float) -> SpatialResult:
    r"""OLS covariance ``sigma^2 (Z'Z)^{-1}`` of the SLX model, ``Z = [X, W X_*]``.

    ``X_*`` are the non-constant columns of ``X`` (their spatial lags enter
    the SLX model; Halleck Vega and Elhorst 2015); the covariance of
    ``(beta, theta)`` given the error variance ``sigma2`` is
    ``sigma2 (Z'Z)^{-1}`` as in :func:`~morie.fn.spslx.slx_regression`.
    ``statistic`` is its trace.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> X = [[1.0, 0.2], [1.0, 0.9], [1.0, 0.4], [1.0, 0.7]]
    >>> round(slxvar(X, W, 1.0).extra["cov"][2][2], 6)
    116.0
    """
    Xm = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    Wm = [[float(v) for v in r] for r in np.asarray(W, dtype=float).tolist()]
    n, p = len(Xm), len(Xm[0])
    cols = [list(c) for c in zip(*Xm)]
    lag = [k for k in range(p) if max(cols[k]) - min(cols[k]) > 0.0]
    Z = [Xm[i] + [ssum(Wm[i][j] * cols[k][j] for j in range(n)) for k in lag] for i in range(n)]
    q = len(Z[0])
    Zc = [list(c) for c in zip(*Z)]
    V = [
        [float(v) * float(sigma2) for v in r]
        for r in inverse([[ssum(a * b for a, b in zip(Zc[i], Zc[j])) for j in range(q)] for i in range(q)])
    ]
    return SpatialResult(
        name="slxvar",
        statistic=ssum(V[k][k] for k in range(q)),
        p_value=None,
        extra={"cov": V, "se": [V[k][k] ** 0.5 for k in range(q)], "lagged_columns": lag},
    )


slxvar_fn = slxvar


def cheatsheet() -> str:
    return "slxvar(X, W, sigma2) -> covariance of the SLX OLS estimator."
