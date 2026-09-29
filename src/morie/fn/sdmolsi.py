# morie.fn -- function file (rootcoder007/morie)
"""SDM OLS ignoring spatial component (baseline)."""

from __future__ import annotations

import math

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmolsi(y, X):
    r"""SDM OLS ignoring spatial component (baseline).

    OLS ``beta = (X'X)^{-1} X'y`` with the ML error variance ``sigma^2 =
    e'e / n`` and the Gaussian log-likelihood ``-n/2 (log(2 pi sigma^2) +
    1)``: the restricted (``rho = 0``, ``theta = 0``) baseline that the
    spatial Durbin model's likelihood-ratio tests compare against
    (Elhorst 2014, sec. 2.3).

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix including any intercept column.

    Returns
    -------
    SpatialResult
        ``statistic`` is the log-likelihood; ``extra`` has ``beta``,
        ``sigma2``, ``residuals`` and ``aic`` (``p + 1`` parameters).

    References
    ----------
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*.
    Springer, Berlin.

    Examples
    --------
    >>> r = sdmolsi([1.0, 2.1, 2.9, 4.2], [[1, 0], [1, 1], [1, 2], [1, 3]])
    >>> [round(b, 12) for b in r.extra["beta"]], round(r.statistic, 12)
    ([0.99, 1.04], 3.437005910819)
    """
    yv, Xm = sd._vec(y), sd._mat(X)
    n, p = len(yv), len(Xm[0])
    XtX = [[sd.ssum(r[a] * r[b] for r in Xm) for b in range(p)] for a in range(p)]
    beta = sd.solve(XtX, [sd.ssum(r[a] * t for r, t in zip(Xm, yv)) for a in range(p)])
    e = [t - sd.ssum(u * v for u, v in zip(r, beta)) for r, t in zip(Xm, yv)]
    s2 = sd.ssum(v * v for v in e) / n
    ll = -0.5 * n * (math.log(2 * math.pi * s2) + 1.0)
    return SpatialResult(
        name="sdmolsi",
        statistic=ll,
        extra={"beta": list(beta), "sigma2": s2, "residuals": e, "aic": -2 * ll + 2 * (p + 1)},
    )


sdmolsi_fn = sdmolsi


def cheatsheet() -> str:
    return "sdmolsi(y, X) -> OLS baseline beta, sigma2, log-likelihood"
