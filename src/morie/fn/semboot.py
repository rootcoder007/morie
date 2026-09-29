# morie.fn -- function file (rootcoder007/morie)
"""SEM bootstrap CI for lambda."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semboot(y, X, W, B=99, seed=0, level=0.95):
    r"""SEM bootstrap CI for lambda.

    Residual bootstrap of the spatial error model (Efron and Tibshirani 1993, ch. 9; for
    spatial models Anselin 1988, sec. 13): fit by maximum likelihood
    (profile likelihood, Brent's method on ``(-0.999, 0.999)``), take the
    innovations ``e = B(Ay - X beta)`` (``A = I - rho W``, ``B = I - lambda
    W``) centred at their mean, resample them with replacement using Philox
    uniforms (index ``floor(u n)``), rebuild ``y* = A^{-1}(X beta +
    B^{-1} e*)``, refit, and take the type-7 quantiles ``(1 - level) / 2``
    and ``(1 + level) / 2`` of the ``B`` refitted ``lambda``.

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
        ``statistic`` is the ML estimate of ``lambda``; ``extra`` has
        ``ci_lower``, ``ci_upper``, ``se_boot`` and ``draws``.

    References
    ----------
    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = semboot([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], X, W, B=9)
    >>> round(r.statistic, 8), r.extra["ci_lower"] <= r.extra["ci_upper"]
    (0.1641051, True)
    """
    est, draws, level = sd.bootstrap(y, X, W, "error", int(B), seed, float(level))
    s = sd.ci_summary(est[2], [d[1] for d in draws], level)
    return SpatialResult(name="semboot", statistic=s.pop("estimate"), extra=s)


semboot_fn = semboot


def cheatsheet() -> str:
    return "semboot(y, X, W, B, seed, level) -> residual-bootstrap CI for lambda"
