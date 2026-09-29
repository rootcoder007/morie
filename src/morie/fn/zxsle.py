# morie.fn -- function file (rootcoder007/morie)
"""Spatial elastic net"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_elastic(y, X, coords, lam, alpha=0.5, *, trend=2):
    r"""Spatial elastic net

    Penalised least squares on the spatial features ``[X, s1, s2, s1^2, s1 s2,
    s2^2]`` (a quadratic trend surface; ``trend=1`` keeps the linear terms),
    ``min 1/(2n) ||y - b0 - Z b||^2 + lam (alpha ||b||_1 + (1 - alpha)
    ||b||^2 / 2)`` on standardised columns (population sd, as glmnet), by
    cyclic coordinate descent to ``1e-12`` (Friedman, Hastie and Tibshirani
    2010); coefficients are returned on the original scale.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    lam : float
        Penalty.
    alpha : float
        Lasso share of the penalty.
    trend : int
        Trend-surface degree.

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of non-zero slopes; ``extra`` has ``coefficients`` (intercept first), ``fitted`` and ``rmse``.

    References
    ----------
    Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS B*,
    67(2), 301-320.

    Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized linear
    models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

    Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, sec. 4.1 (trend surfaces).

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> r = spatial_elastic(y, X, S, 0.05)
    >>> [round(b, 10) for b in r.extra["coefficients"][:3]]
    [0.9377264027, 1.8771575483, 1.1341041793]
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, trend)
    b = sm.coord_descent(Z, yv, float(lam), float(alpha))
    fit = sm.ols_predict(b, Z)
    return DescriptiveResult(
        name="zxsle",
        value=float(sum(1 for v in b[1:] if v != 0.0)),
        extra={"coefficients": b, "fitted": fit, "rmse": sm.rmse(yv, fit)},
    )


spat = spatial_elastic
spatialelastic = spatial_elastic


def cheatsheet() -> str:
    return "spatial_elastic(...) -> Spatial elastic net"
