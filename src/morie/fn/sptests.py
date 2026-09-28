# morie.fn -- function file (rootcoder007/morie)
"""Spatial Durbin model by maximum likelihood with spillover indices, and Breusch-Pagan tests
(original and Koenker's studentized) for OLS and spatial-model residuals."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rrng_core import pchisq
from .sarreg import spatial_regression_ml
from .spslx import spatial_impacts

__all__ = ["sdm_ml", "bp_test", "spatial_bp_test"]


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _mv(W, v):
    return [ssum(a * b for a, b in zip(r, v)) for r in W]


def sdm_ml(y, X, W, *, interval=(-0.999, 0.999)) -> RichResult:
    r"""Spatial Durbin model ``y = rho W y + X beta + W X theta + e`` by maximum likelihood, as ``spatialreg::lagsarlm(Durbin = TRUE)``.

    The lag model of :func:`~morie.fn.sarreg.spatial_regression_ml` on ``[X,
    W X]`` (constant columns of ``X`` are not lagged); average direct,
    indirect and total impacts (LeSage and Pace 2009) of the non-constant
    regressors and their spillover index ``indirect / total``.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> r = sdm_ml([1.0, 2.5, 1.5, 3.0, ], [[1, 0.1], [1, 0.7], [1, 0.2], [1, 0.9]], W)
    >>> len(r.beta), len(r.theta)
    (2, 1)
    """
    Xm = _mat(X)
    Wm = _mat(W)
    n, p = len(Xm), len(Xm[0])
    lagged = [u for u in range(p) if any(r[u] != Xm[0][u] for r in Xm)]
    WX = [_mv(Wm, [r[u] for r in Xm]) for u in lagged]
    Z = [Xm[i] + [WX[k][i] for k in range(len(lagged))] for i in range(n)]
    fit = spatial_regression_ml(y, Z, Wm, model="lag", interval=interval)
    coef = [float(v) for v in fit.value]
    beta, theta = coef[:p], coef[p:]
    rho = fit.extra["rho"]
    imp = spatial_impacts(rho, [beta[u] for u in lagged], Wm, theta=theta)
    return RichResult(
        payload={
            "beta": beta,
            "theta": theta,
            "rho": rho,
            "sigma2": fit.extra["sigma2"],
            "loglik": fit.extra["loglik"],
            "aic": fit.extra["aic"],
            "se": fit.extra["se"],
            "direct": imp.direct,
            "indirect": imp.indirect,
            "total": imp.total,
            "spillover_index": [a / t if t else math.nan for a, t in zip(imp.indirect, imp.total)],
            "lagged": lagged,
        }
    )


def bp_test(residuals, Z, *, studentize: bool = True) -> RichResult:
    r"""Breusch-Pagan test of residual variance depending on ``Z`` (intercept included), as ``lmtest::bptest`` / ``spatialreg::bptest.Sarlm``.

    With ``sigma^2 = sum e^2 / n``: original (Breusch and Pagan 1979)
    ``BP = 0.5 ESS`` of the regression of ``e^2 / sigma^2 - 1`` on ``Z``;
    studentized (Koenker 1981) ``BP = n ESS / TSS`` of the regression of
    ``e^2 - sigma^2`` on ``Z`` (``n R^2``). Chi-square with ``k - 1`` df.

    References
    ----------
    Breusch, T. S. and Pagan, A. R. (1979). A simple test for
    heteroscedasticity and random coefficient variation. *Econometrica*,
    47(5), 1287-1294.
    Koenker, R. (1981). A note on studentizing a test for heteroscedasticity.
    *Journal of Econometrics*, 17(1), 107-112.

    Examples
    --------
    >>> r = bp_test([1, -1, 2, -2, 3, -3], [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]])
    >>> round(r.statistic, 12), r.df
    (6.0, 2)
    """
    e = [float(v) for v in residuals]
    Zm = _mat(Z)
    n, k = len(Zm), len(Zm[0])
    s2 = ssum(v * v for v in e) / n
    w = [v * v - s2 for v in e] if studentize else [v * v / s2 - 1 for v in e]
    ZtZ = inverse([[ssum(Zm[i][a] * Zm[i][b] for i in range(n)) for b in range(k)] for a in range(k)])
    Ztw = [ssum(Zm[i][a] * w[i] for i in range(n)) for a in range(k)]
    g = [ssum(float(ZtZ[a][b]) * Ztw[b] for b in range(k)) for a in range(k)]
    fv = [ssum(Zm[i][a] * g[a] for a in range(k)) for i in range(n)]
    bp = n * ssum(v * v for v in fv) / ssum(v * v for v in w) if studentize else 0.5 * ssum(v * v for v in fv)
    return RichResult(
        payload={
            "statistic": bp,
            "df": k - 1,
            "p_value": float(pchisq(bp, k - 1, lower_tail=False)),
            "studentize": studentize,
        }
    )


def spatial_bp_test(y, X, W, *, model: str = "lag", studentize: bool = True) -> RichResult:
    r"""Breusch-Pagan test on the residuals of a spatial lag or error model fitted by ML (as ``spatialreg::bptest.Sarlm``).

    Lag: residuals ``y - rho W y - X beta`` and ``Z = X``; error: residuals
    ``(I - lambda W)(y - X beta)`` and the filtered ``Z = X - lambda W X``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> r = spatial_bp_test([1.0, 2.5, 1.5, 3.0, ], [[1, 0.1], [1, 0.7], [1, 0.2], [1, 0.9]], W)
    >>> r.df
    1
    """
    if model not in ("lag", "error"):
        raise ValueError("model must be lag or error")
    yv = [float(v) for v in y]
    Xm = _mat(X)
    Wm = _mat(W)
    n, p = len(Xm), len(Xm[0])
    fit = spatial_regression_ml(yv, Xm, Wm, model=model)
    b = [float(v) for v in fit.value]
    Wy = _mv(Wm, yv)
    if model == "lag":
        rho = fit.extra["rho"]
        e = [yv[i] - rho * Wy[i] - ssum(Xm[i][u] * b[u] for u in range(p)) for i in range(n)]
        Z = Xm
    else:
        lam = fit.extra["lambda"]
        WX = [_mv(Wm, [r[u] for r in Xm]) for u in range(p)]
        Z = [[Xm[i][u] - lam * WX[u][i] for u in range(p)] for i in range(n)]
        e = [yv[i] - lam * Wy[i] - ssum(Z[i][u] * b[u] for u in range(p)) for i in range(n)]
    bp = bp_test(e, Z, studentize=studentize)
    return RichResult(
        payload={
            "statistic": bp.statistic,
            "df": bp.df,
            "p_value": bp.p_value,
            "studentize": studentize,
            "residuals": e,
            "model": model,
        }
    )


def cheatsheet() -> str:
    return "sdm_ml / bp_test / spatial_bp_test -> spatial Durbin ML with spillovers; Breusch-Pagan tests."
