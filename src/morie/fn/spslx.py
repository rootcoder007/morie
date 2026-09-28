# morie.fn -- function file (rootcoder007/morie)
"""SLX regression and direct/indirect/total impacts of spatial lag and Durbin models (LeSage and Pace 2009)."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["slx_regression", "spatial_impacts"]


def _mat(A):
    return [[float(v) for v in r] for r in np.asarray(A, dtype=float).tolist()]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[ssum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def slx_regression(y, X, W) -> RichResult:
    r"""Spatial lag of X (SLX) model by OLS, with impacts and a Wald test.

    ``y = X beta + W X_* theta + e`` where ``X_*`` are the non-constant
    columns of ``X``, estimated by OLS as ``spatialreg::lmSLX``; ``sigma^2 =
    e'e / (n - k)``.  The impacts of covariate ``r`` are direct ``beta_r``,
    indirect ``theta_r`` and total ``beta_r + theta_r`` (LeSage and Pace
    2009, sec. 2.7; Halleck Vega and Elhorst 2015) with standard errors from
    the OLS covariance; the Wald statistic ``theta' V_theta^{-1} theta`` tests
    ``theta = 0`` against chi-square with ``k_*`` df.

    :param y: Response (n,).
    :param X: Design matrix (n, p) including any intercept column.
    :param W: Spatial weights (n, n).
    :return: :class:`RichResult` with ``coefficients`` (X then WX),
        ``se``, ``cov``, ``sigma2``, ``residuals``, ``impacts`` (per
        lagged covariate: direct, indirect, total and their ``se``),
        ``wald`` and ``wald_p``.

    References
    ----------
    Halleck Vega, S. and Elhorst, J. P. (2015). The SLX model. *Journal of
    Regional Science*, 55(3), 339-363.
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> r = slx_regression([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W)
    >>> [round(v, 6) for v in r.coefficients]
    [1.287716, 2.116493, -0.951624]
    """
    yv = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    Xm, Wm = _mat(X), _mat(W)
    n, p = len(yv), len(Xm[0])
    cols = [list(c) for c in zip(*Xm)]
    lag = [k for k in range(p) if max(cols[k]) - min(cols[k]) > 0.0]
    WX = [[ssum(Wm[i][j] * cols[k][j] for j in range(n)) for k in lag] for i in range(n)]
    Z = [Xm[i] + WX[i] for i in range(n)]
    q = len(Z[0])
    Zc = [list(c) for c in zip(*Z)]
    ZZi = [
        [float(v) for v in r]
        for r in inverse([[ssum(a * b for a, b in zip(Zc[i], Zc[j])) for j in range(q)] for i in range(q)])
    ]
    Zy = [ssum(a * b for a, b in zip(c, yv)) for c in Zc]
    b = [ssum(ZZi[i][j] * Zy[j] for j in range(q)) for i in range(q)]
    e = [yv[i] - ssum(Z[i][k] * b[k] for k in range(q)) for i in range(n)]
    s2 = ssum(v * v for v in e) / (n - q)
    V = [[v * s2 for v in r] for r in ZZi]
    se = [V[k][k] ** 0.5 for k in range(q)]
    imp = []
    for t, k in enumerate(lag):
        th = p + t
        tot_var = V[k][k] + V[th][th] + 2.0 * V[k][th]
        imp.append(
            {
                "column": k,
                "direct": b[k],
                "indirect": b[th],
                "total": b[k] + b[th],
                "se_direct": se[k],
                "se_indirect": se[th],
                "se_total": tot_var**0.5,
            }
        )
    idx = list(range(p, q))
    Vt = [[V[i][j] for j in idx] for i in idx]
    th = [b[i] for i in idx]
    Vti = [[float(v) for v in r] for r in inverse(Vt)] if idx else []
    wald = ssum(th[i] * Vti[i][j] * th[j] for i in range(len(idx)) for j in range(len(idx))) if idx else 0.0
    return RichResult(
        payload={
            "coefficients": b,
            "se": se,
            "cov": V,
            "sigma2": s2,
            "residuals": e,
            "impacts": imp,
            "wald": wald,
            "wald_p": float(stats.chi2.sf(wald, len(idx))) if idx else 1.0,
            "lagged_columns": lag,
        }
    )


def spatial_impacts(rho: float, beta, W, theta=None) -> RichResult:
    r"""Average direct, indirect and total impacts of a spatial lag or Durbin model.

    With ``S_r(W) = (I - rho W)^{-1} (beta_r I + theta_r W)`` (``theta = 0``
    for the lag model), the average direct impact is ``tr(S_r)/n``, the total
    ``1' S_r 1 / n`` and the indirect their difference (LeSage and Pace 2009,
    sec. 2.7), exactly ``spatialreg::impacts`` with ``listw`` (exact
    inverse).  ``beta`` and ``theta`` exclude the intercept.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = spatial_impacts(0.4, [2.0], W)
    >>> [round(r.direct[0], 6), round(r.total[0], 6)]
    [2.253968, 3.333333]
    """
    Wm = _mat(W)
    n = len(Wm)
    bet = [float(v) for v in np.asarray(beta, dtype=float).ravel().tolist()]
    tht = [0.0] * len(bet) if theta is None else [float(v) for v in np.asarray(theta, dtype=float).ravel().tolist()]
    if len(tht) != len(bet):
        raise ValueError("theta must match beta")
    A = [[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(n)] for i in range(n)]
    Ai = [[float(v) for v in r] for r in inverse(A)]
    AiW = _mm(Ai, Wm)
    trA, sA = ssum(Ai[i][i] for i in range(n)), ssum(v for r in Ai for v in r)
    trAW, sAW = ssum(AiW[i][i] for i in range(n)), ssum(v for r in AiW for v in r)
    direct = [(bet[r] * trA + tht[r] * trAW) / n for r in range(len(bet))]
    total = [(bet[r] * sA + tht[r] * sAW) / n for r in range(len(bet))]
    return RichResult(payload={"direct": direct, "indirect": [t - d for t, d in zip(total, direct)], "total": total})


def cheatsheet() -> str:
    return "slx_regression(y, X, W) / spatial_impacts(rho, beta, W, theta) -> SLX fit and LeSage-Pace impacts."
