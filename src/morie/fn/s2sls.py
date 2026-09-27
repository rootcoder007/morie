# morie.fn -- function file (rootcoder007/morie)
"""Spatial two-stage least squares for the spatial lag model (Kelejian and Prucha 1998)."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import inverse, ssum


def _mv(W, v):
    return [ssum(a * b for a, b in zip(row, v)) for row in W]


def _xtx(A, B):
    return [[ssum(r[a] * s[b] for r, s in zip(A, B)) for b in range(len(B[0]))] for a in range(len(A[0]))]


def spatial_two_stage_least_squares(
    y, X, W, *, w2x: bool = True, robust=None, sig2n_k: bool = True
) -> DescriptiveResult:
    """Spatial two-stage least squares for ``y = rho W y + X beta + e``.

    ``Wy`` is endogenous; it is instrumented by ``X`` and the spatial lags
    ``W X`` (and ``W^2 X`` when ``w2x``) of the non-constant columns of
    ``X``, plus ``W 1`` and ``W^2 1`` for an intercept when ``W`` is not
    row-standardised (Kelejian and Prucha 1998); a lag that is constant
    (``W 1`` when every row has the same sum) duplicates the intercept and is
    dropped, where ``spatialreg`` keeps it and returns a degenerate fit. With
    ``Q`` the instrument
    matrix, ``Wy_hat = Q (Q'Q)^-1 Q' W y``, ``Zp = [Wy_hat, X]`` and
    ``Z = [Wy, X]``: ``delta = (Zp'Zp)^-1 Zp' y``, residuals ``e = y - Z
    delta``, and ``Var = s^2 (Zp'Zp)^-1`` with ``s^2 = e'e / (n - k)`` (or
    ``/ n`` without ``sig2n_k``); ``robust = "HC0"`` or ``"HC1"`` gives the
    heteroskedasticity-consistent sandwich. This is
    ``spatialreg::stsls``.

    :param y: Response (n).
    :param X: Design matrix (n x p), intercept (if any) in any column.
    :param W: Spatial weights (n x n).
    :param w2x: Also use ``W^2 X`` as instruments.
    :param robust: ``None``, ``"HC0"`` or ``"HC1"``.
    :param sig2n_k: Divide the residual sum of squares by ``n - k``.
    :return: DescriptiveResult; ``value`` is ``[rho, beta...]``; ``extra``
        has ``se``, ``cov``, ``sigma2``, ``residuals``, ``instruments``
        (column count).

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial
    two-stage least squares procedure for estimating a spatial
    autoregressive model with autoregressive disturbances. Journal of Real
    Estate Finance and Economics 17, 99-121.

    Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(6)] for i in range(6)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5)]
    >>> r = spatial_two_stage_least_squares([1.0, 2.2, 1.4, 3.1, 0.9, 2.0], X, W)
    >>> len(r.value), r.extra["instruments"]
    (3, 2)
    """
    y = [float(v) for v in y]
    X = [[float(v) for v in r] for r in X]
    W = [[float(v) for v in r] for r in W]
    n, p = len(y), len(X[0])
    const = [k for k in range(p) if all(r[k] == X[0][k] for r in X)]
    rowstd = all(abs(ssum(r) - 1.0) < 1e-12 for r in W)
    inst = []
    for k in range(p):
        if k in const and rowstd:
            continue
        col = [r[k] for r in X]
        wc = _mv(W, col)
        for c in [wc, _mv(W, wc)] if w2x else [wc]:
            # a constant lag (e.g. W 1 when all rows share one sum) repeats the intercept
            if not (const and max(c) - min(c) <= 1e-12 * max(1.0, abs(c[0]))):
                inst.append(c)
    Q = [X[i] + [c[i] for c in inst] for i in range(n)]
    Wy = _mv(W, y)
    QtQi = inverse(_xtx(Q, Q))
    Qty = [ssum(Q[i][a] * Wy[i] for i in range(n)) for a in range(len(Q[0]))]
    bz = [ssum(QtQi[a][b] * Qty[b] for b in range(len(Qty))) for a in range(len(Qty))]
    yhat = [ssum(Q[i][a] * bz[a] for a in range(len(bz))) for i in range(n)]
    Zp = [[yhat[i]] + X[i] for i in range(n)]
    Z = [[Wy[i]] + X[i] for i in range(n)]
    ZZi = inverse(_xtx(Zp, Zp))
    Zy = [ssum(Zp[i][a] * y[i] for i in range(n)) for a in range(p + 1)]
    delta = [ssum(ZZi[a][b] * Zy[b] for b in range(p + 1)) for a in range(p + 1)]
    e = [y[i] - ssum(Z[i][a] * delta[a] for a in range(p + 1)) for i in range(n)]
    sse = ssum(t * t for t in e)
    df = n - (p + 1) if sig2n_k else n
    if robust is None:
        cov = [[v * sse / df for v in row] for row in ZZi]
    else:
        om = [t * t * (n / df if robust == "HC1" else 1.0) for t in e]
        M = [[ssum(Zp[i][a] * Zp[i][b] * om[i] for i in range(n)) for b in range(p + 1)] for a in range(p + 1)]
        cov = [
            [ssum(ZZi[a][u] * M[u][v] * ZZi[v][b] for u in range(p + 1) for v in range(p + 1)) for b in range(p + 1)]
            for a in range(p + 1)
        ]
    return DescriptiveResult(
        name="spatial_two_stage_least_squares",
        value=delta,
        extra={
            "se": [math.sqrt(cov[a][a]) for a in range(p + 1)],
            "cov": cov,
            "sigma2": sse / df,
            "residuals": e,
            "instruments": len(inst),
        },
    )


s2sls = spatial_two_stage_least_squares


def cheatsheet() -> str:
    return "spatial_two_stage_least_squares(y, X, W) -> Kelejian-Prucha spatial 2SLS for the lag model (= spatialreg::stsls)"
