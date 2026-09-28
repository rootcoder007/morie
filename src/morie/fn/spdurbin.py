# morie.fn -- function file (rootcoder007/morie)
"""Spatial Durbin error (SDEM), general nesting (GNS) and conditional autoregressive (CAR) regression by
maximum likelihood, with impacts, likelihood-ratio and Wald tests, the exact residual Moran test and the
log-Jacobian (spatialreg conventions)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rrng_core import pchisq, pnorm
from .sarreg import _brent, _logdet, _mv, spatial_regression_ml
from .spslx import spatial_impacts

__all__ = ["sdem_ml", "gns_ml", "car_ml", "spatial_lr_test", "spatial_wald_test", "residual_moran", "log_jacobian"]


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _durbin(X, W):
    """[X | W X_*] with X_* the non-constant columns (spatialreg ``Durbin = TRUE``)."""
    p = len(X[0])
    lag = [k for k in range(p) if len({r[k] for r in X}) > 1]
    WX = [_mv(W, [r[k] for r in X]) for k in lag]
    return [list(r) + [WX[j][i] for j in range(len(lag))] for i, r in enumerate(X)], lag


def _cov_beta(Z, W, lam, s2):
    n, q = len(Z), len(Z[0])
    cols = [[r[k] for r in Z] for k in range(q)]
    WZ = [_mv(W, c) for c in cols]
    Zs = [[cols[k][i] - lam * WZ[k][i] for k in range(q)] for i in range(n)]
    return [
        [v * s2 for v in row] for row in inverse([[ssum(r[a] * r[b] for r in Zs) for b in range(q)] for a in range(q)])
    ]


def sdem_ml(y, X, W, *, interval=(-0.999, 0.999)) -> RichResult:
    r"""Spatial Durbin error model ``y = X beta + W X_* theta + u``, ``u = lambda W u + e``, by ML.

    Fitted as the spatial error model on ``[X, W X_*]`` (``X_*`` the
    non-constant columns) with :func:`morie.fn.sarreg.spatial_regression_ml`,
    as ``spatialreg::errorsarlm(Durbin = TRUE)``. The coefficient covariance
    is ``sigma^2 (Z_*' Z_*)^{-1}``, ``Z_* = (I - lambda W) Z``. Impacts are
    direct ``beta_r``, indirect ``theta_r`` and total ``beta_r + theta_r`` with
    standard errors from that covariance (LeSage and Pace 2009; Halleck Vega
    and Elhorst 2015).

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.
    Halleck Vega, S. and Elhorst, J. P. (2015). The SLX model. *Journal of
    Regional Science*, 55(3), 339-363.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> X = [[1, 0.1], [1, 0.5], [1, 0.2], [1, 0.9]]
    >>> r = sdem_ml([1.0, 2.0, 1.5, 3.0], X, W)
    >>> len(r.coefficients), r.impacts["total"][0] == r.coefficients[1] + r.coefficients[2]
    (3, True)
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm, Wm = _mat(X), _mat(W)
    Z, lag = _durbin(Xm, Wm)
    r = spatial_regression_ml(yv, Z, Wm, model="error", interval=interval)
    b, lam, s2 = list(r.value), r.extra["lambda"], r.extra["sigma2"]
    V = _cov_beta(Z, Wm, lam, s2)
    p = len(Xm[0])
    d, ind = [b[k] for k in lag], [b[p + j] for j in range(len(lag))]
    se_tot = [math.sqrt(V[k][k] + V[p + j][p + j] + 2 * V[k][p + j]) for j, k in enumerate(lag)]
    return RichResult(
        payload={
            "coefficients": b,
            "se": r.extra["se"],
            "cov": V,
            "lambda": lam,
            "sigma2": s2,
            "loglik": r.extra["loglik"],
            "aic": r.extra["aic"],
            "bic": r.extra["bic"],
            "lagged": lag,
            "impacts": {
                "direct": d,
                "indirect": ind,
                "total": [u + v for u, v in zip(d, ind)],
                "se_direct": [math.sqrt(V[k][k]) for k in lag],
                "se_indirect": [math.sqrt(V[p + j][p + j]) for j in range(len(lag))],
                "se_total": se_tot,
            },
        }
    )


def gns_ml(y, X, W, *, interval=(-0.999, 0.999)) -> RichResult:
    r"""General nesting spatial model ``y = rho W y + X beta + W X_* theta + u``, ``u = lambda W u + e``, by ML.

    Fitted as the SAC model on ``[X, W X_*]`` (``spatialreg::sacsarlm(Durbin
    = TRUE)``; Elhorst 2014). Impacts use ``S_r = (I - rho W)^{-1} (beta_r I
    + theta_r W)`` (:func:`morie.fn.spslx.spatial_impacts`).

    References
    ----------
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data
    to Spatial Panels*. Springer.

    Examples
    --------
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> r = gns_ml([1.0, 2.0, 1.5, 3.0], [[1, 0.1], [1, 0.5], [1, 0.2], [1, 0.9]], W)
    >>> sorted(r.impacts)
    ['direct', 'indirect', 'total']
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm, Wm = _mat(X), _mat(W)
    Z, lag = _durbin(Xm, Wm)
    r = spatial_regression_ml(yv, Z, Wm, model="sac", interval=interval)
    b = list(r.value)
    p = len(Xm[0])
    imp = spatial_impacts(r.extra["rho"], [b[k] for k in lag], Wm, [b[p + j] for j in range(len(lag))])
    return RichResult(
        payload={
            "coefficients": b,
            "se": r.extra["se"],
            "rho": r.extra["rho"],
            "lambda": r.extra["lambda"],
            "sigma2": r.extra["sigma2"],
            "loglik": r.extra["loglik"],
            "aic": r.extra["aic"],
            "bic": r.extra["bic"],
            "lagged": lag,
            "impacts": {k: list(imp[k]) for k in ("direct", "indirect", "total")},
        }
    )


def car_ml(y, X, W, *, interval=None) -> RichResult:
    r"""Conditional autoregressive (CAR) regression by ML, as ``spatialreg::spautolm(family = "CAR")``.

    ``y = X beta + u``, ``u ~ N(0, sigma^2 (I - lambda W)^{-1})`` with ``W``
    symmetric. ``log L = -n/2 log(2 pi sigma^2) + 1/2 log|I - lambda W| - r'
    (I - lambda W) r / (2 sigma^2)``; profiling ``beta = (X'AX)^{-1} X'Ay``
    and ``sigma^2 = r'Ar/n`` leaves a one-dimensional search (Brent) over
    ``interval``, by default ``(1/e_min, 1/e_max)`` from the eigenvalues of
    ``W``. Also reports ``beta`` standard errors ``sqrt(diag(sigma^2
    (X'AX)^{-1}))`` and the LR test of ``lambda = 0`` (Besag 1974; Cressie
    1993).

    References
    ----------
    Besag, J. (1974). Spatial interaction and the statistical analysis of
    lattice systems. *Journal of the Royal Statistical Society B*, 36(2),
    192-236.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = car_ml([1.0, 2.2, 2.9, 4.1], [[1, 0.0], [1, 1.0], [1, 2.0], [1, 3.0]], W)
    >>> r.lr_test["df"]
    1
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm, Wm = _mat(X), _mat(W)
    n, p = len(yv), len(Xm[0])
    if interval is None:
        ev = [float(v) for v in np.linalg.eigvalsh(np.asarray(Wm, dtype=float)).tolist()]
        interval = (1 / min(ev), 1 / max(ev))

    def prof(lam):
        WX = [_mv(Wm, [r[k] for r in Xm]) for k in range(p)]
        AX = [[Xm[i][k] - lam * WX[k][i] for k in range(p)] for i in range(n)]
        Wy = _mv(Wm, yv)
        Ay = [a - lam * b for a, b in zip(yv, Wy)]
        XtAX = [[ssum(Xm[i][a] * AX[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        b = solve(XtAX, [ssum(Xm[i][a] * Ay[i] for i in range(n)) for a in range(p)])
        res = [t - ssum(u * v for u, v in zip(r, b)) for r, t in zip(Xm, yv)]
        Wr = _mv(Wm, res)
        s2 = ssum(e * (e - lam * w) for e, w in zip(res, Wr)) / n
        return b, s2, XtAX

    def nll(lam):
        _, s2, _ = prof(lam)
        return 0.5 * n * math.log(2 * math.pi * s2) + 0.5 * n - 0.5 * _logdet(Wm, lam)

    lam, f = _brent(nll, interval[0], interval[1])
    b, s2, XtAX = prof(lam)
    Vb = [[v * s2 for v in row] for row in inverse(XtAX)]
    ll0 = -nll(0.0)
    lr = 2 * (-f - ll0)
    return RichResult(
        payload={
            "coefficients": b,
            "se": [math.sqrt(Vb[k][k]) for k in range(p)],
            "lambda": lam,
            "sigma2": s2,
            "loglik": -f,
            "aic": 2 * f + 2 * (p + 2),
            "lr_test": {"statistic": lr, "df": 1, "pvalue": 1 - float(pchisq(lr, 1))},
        }
    )


def spatial_lr_test(ll_full: float, ll_restricted: float, df: int) -> RichResult:
    r"""Likelihood-ratio test ``LR = 2 (l_full - l_restricted)`` against chi-square with ``df`` degrees of freedom.

    E.g. GNS vs SDM (``df = 1``), SDEM vs SEM (``df`` = number of lagged
    covariates), the common-factor test SDM vs SEM, or CAR vs OLS.

    Examples
    --------
    >>> r = spatial_lr_test(-10.0, -12.5, 1)
    >>> r.statistic, round(r.pvalue, 6)
    (5.0, 0.025347)
    """
    lr = 2 * (float(ll_full) - float(ll_restricted))
    return RichResult(payload={"statistic": lr, "df": int(df), "pvalue": 1 - float(pchisq(lr, int(df)))})


def spatial_wald_test(estimate, vcov) -> RichResult:
    r"""Wald test ``b' V^{-1} b`` of ``b = 0`` against chi-square with ``len(b)`` degrees of freedom.

    Examples
    --------
    >>> r = spatial_wald_test([0.3], [[0.01]])
    >>> round(r.statistic, 6), round(r.pvalue, 6)
    (9.0, 0.0027)
    """
    b = [float(v) for v in (estimate.tolist() if hasattr(estimate, "tolist") else estimate)]
    V = _mat(vcov)
    w = ssum(u * v for u, v in zip(b, solve(V, b)))
    return RichResult(payload={"statistic": w, "df": len(b), "pvalue": 1 - float(pchisq(w, len(b)))})


def residual_moran(y, X, W) -> RichResult:
    r"""Moran's I of OLS residuals with its exact moments under normality, as ``spdep::lm.morantest``.

    ``I = (n / S0) e'We / e'e``; with ``M = I - X(X'X)^{-1}X'``, ``E(I) = (n
    / S0) tr(MW) / (n - k)`` and ``Var(I) = (n/S0)^2 [tr(MWMW') + tr((MW)^2)
    + tr(MW)^2] / ((n - k)(n - k + 2)) - E(I)^2`` (Cliff and Ord 1981); the
    p-value is one-sided (greater).

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = residual_moran([1.0, 2.0, 2.5, 4.5], [[1, 0.0], [1, 1.0], [1, 2.0], [1, 3.0]], W)
    >>> round(r.expected, 6)
    -0.7
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm, Wm = _mat(X), _mat(W)
    n, k = len(yv), len(Xm[0])
    XtXi = inverse([[ssum(r[a] * r[b] for r in Xm) for b in range(k)] for a in range(k)])
    H = [
        [ssum(Xm[i][a] * XtXi[a][b] * Xm[j][b] for a in range(k) for b in range(k)) for j in range(n)] for i in range(n)
    ]
    M = [[(1.0 if i == j else 0.0) - H[i][j] for j in range(n)] for i in range(n)]
    e = [ssum(M[i][j] * yv[j] for j in range(n)) for i in range(n)]
    S0 = ssum(v for r in Wm for v in r)
    moran_i = n / S0 * ssum(a * b for a, b in zip(e, _mv(Wm, e))) / ssum(a * a for a in e)
    MW = [[ssum(M[i][m] * Wm[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
    trMW = ssum(MW[i][i] for i in range(n))
    MWt = [[ssum(M[i][m] * Wm[j][m] for m in range(n)) for j in range(n)] for i in range(n)]
    tr1 = ssum(MW[i][m] * MWt[m][i] for i in range(n) for m in range(n))
    tr2 = ssum(MW[i][m] * MW[m][i] for i in range(n) for m in range(n))
    E = n / S0 * trMW / (n - k)
    Var = (n / S0) ** 2 * (tr1 + tr2 + trMW**2) / ((n - k) * (n - k + 2)) - E * E
    z = (moran_i - E) / math.sqrt(Var)
    return RichResult(payload={"I": moran_i, "expected": E, "variance": Var, "z": z, "pvalue": 1 - float(pnorm(z))})


def log_jacobian(W, rho: float, lam: float = 0.0) -> float:
    r"""Log-Jacobian ``log|I - rho W| + log|I - lambda W|`` of spatial lag / error / SAC models (LU).

    Examples
    --------
    >>> round(log_jacobian([[0, 1], [1, 0]], 0.5), 6)
    -0.287682
    """
    Wm = _mat(W)
    return (_logdet(Wm, rho) if rho else 0.0) + (_logdet(Wm, lam) if lam else 0.0)


def cheatsheet() -> str:
    return (
        "sdem_ml / gns_ml / car_ml / spatial_lr_test / spatial_wald_test / residual_moran / log_jacobian -> "
        "spatial Durbin error, general nesting and CAR models with tests."
    )
