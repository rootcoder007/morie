# morie.fn -- function file (rootcoder007/morie)
"""Spatial quantile regression"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult
from .spquant import quantile_regression_lp


def spatial_quantile(y, X, coords, tau=0.5, *, trend=1):
    r"""Spatial quantile regression

    Linear quantile regression (Koenker and Bassett 1978) of ``y`` on the
    spatial features ``[1, X, s1, s2]`` (``trend=2`` adds the quadratic
    trend terms): minimise ``sum rho_tau(y_i - z_i'b)`` by the simplex method,
    :func:`morie.fn.spquant.quantile_regression_lp`.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    tau : float
        Quantile level.
    trend : int
        Trend-surface degree.

    Returns
    -------
    DescriptiveResult
        ``value`` is the check-loss objective; ``extra`` has ``coefficients`` (intercept first) and ``fitted``.

    References
    ----------
    Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.

    Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, sec. 4.1 (trend surfaces).

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> r = spatial_quantile(y, X, S, 0.5)
    >>> round(r.value, 10)
    1.0
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, trend)
    r = quantile_regression_lp(yv, [[1.0] + row for row in Z], tau=float(tau))
    b = [float(v) for v in r["coefficients"]]
    return DescriptiveResult(
        name="zxsqr",
        value=float(r["objective"]),
        extra={"coefficients": b, "fitted": sm.ols_predict(b, Z), "tau": float(tau)},
    )


spat = spatial_quantile


def cheatsheet() -> str:
    return "spatial_quantile(...) -> Spatial quantile regression"
