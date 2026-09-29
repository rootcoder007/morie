# morie.fn -- function file (rootcoder007/morie)
"""SLX bootstrap CI for theta."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from ._rng import random_uniform


def slxboot(y, X, W, B=99, seed=0, level=0.95):
    r"""SLX bootstrap CI for theta.

    Residual bootstrap of the SLX model ``y = X beta + W X_* theta + e``
    fitted by OLS on ``Z = [X, W X_*]`` (Halleck Vega and Elhorst 2015):
    resample the centred OLS residuals with Philox uniforms (index
    ``floor(u n)``), rebuild ``y* = Z g + e*``, refit, and take the type-7
    quantiles of each refitted ``theta_k`` (Efron and Tibshirani 1993,
    ch. 9).

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix including any intercept column.
    W : array-like, shape (n, n)
        Spatial weights.
    B : int
        Bootstrap replicates.
    seed : int
        Philox seed.
    level : float
        Confidence level.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``theta`` of the first lagged column; ``extra``
        has ``theta``, ``ci_lower``, ``ci_upper``, ``se_boot`` (lists over
        the lagged columns), ``lagged_columns`` and ``draws``.

    References
    ----------
    Halleck Vega, S. and Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3),
    339-363.

    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = slxboot([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], X, W, B=9)
    >>> round(r.statistic, 12), round(r.extra["ci_lower"][0], 12)
    (0.625965428466, 0.478961961249)
    """
    yv = sd._vec(y)
    Z, lag = sd.durbin(X, W)
    n, q = len(yv), len(Z[0])
    p = q - len(lag)
    ZtZ = [[sd.ssum(r[a] * r[b] for r in Z) for b in range(q)] for a in range(q)]

    def ols(v):
        return sd.solve(ZtZ, [sd.ssum(r[a] * t for r, t in zip(Z, v)) for a in range(q)])

    g = ols(yv)
    fit = [sd.ssum(u * v for u, v in zip(r, g)) for r in Z]
    e = [a - b for a, b in zip(yv, fit)]
    mu = sd.ssum(e) / n
    e = [v - mu for v in e]
    u = random_uniform(int(B) * n, seed=seed)
    draws = []
    for b in range(int(B)):
        ys = [fit[i] + e[min(int(u[b * n + i] * n), n - 1)] for i in range(n)]
        draws.append(list(ols(ys))[p:])
    theta = list(g)[p:]
    a = (1.0 - float(level)) / 2.0
    cols = [[d[j] for d in draws] for j in range(len(lag))]
    return SpatialResult(
        name="slxboot",
        statistic=theta[0],
        extra={
            "theta": theta,
            "ci_lower": [sd.quantile7(c, a) for c in cols],
            "ci_upper": [sd.quantile7(c, 1.0 - a) for c in cols],
            "se_boot": [sd.sd(c) for c in cols],
            "lagged_columns": lag,
            "draws": draws,
        },
    )


slxboot_fn = slxboot


def cheatsheet() -> str:
    return "slxboot(y, X, W, B, seed, level) -> residual-bootstrap CI for theta"
