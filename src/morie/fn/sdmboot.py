# morie.fn -- function file (rootcoder007/morie)
"""SDM bootstrap CI for rho."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmboot(y, X, W, B=99, seed=0, level=0.95):
    r"""SDM bootstrap CI for rho.

    The spatial Durbin model is the lag model on ``Z = [X, W X_*]``
    (``X_*`` the non-constant columns); the interval is the residual
    bootstrap of that lag model (Efron and Tibshirani 1993, ch. 9): fit by
    ML, resample the centred innovations ``e = (I - rho W) y - Z beta``
    with Philox uniforms, rebuild ``y* = (I - rho W)^{-1}(Z beta + e*)``,
    refit, and take type-7 quantiles of the refitted ``rho``.

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
        ``statistic`` is the ML ``rho``; ``extra`` has ``ci_lower``,
        ``ci_upper``, ``se_boot`` and ``draws``.

    References
    ----------
    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = sdmboot([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], X, W, B=9)
    >>> round(r.statistic, 8), r.extra["ci_lower"] <= r.extra["ci_upper"]
    (-0.60639783, True)
    """
    Z, _lag = sd.durbin(X, W)
    est, draws, level = sd.bootstrap(y, Z, W, "lag", int(B), seed, float(level))
    s = sd.ci_summary(est[1], [d[0] for d in draws], level)
    return SpatialResult(name="sdmboot", statistic=s.pop("estimate"), extra=s)


sdmboot_fn = sdmboot


def cheatsheet() -> str:
    return "sdmboot(y, X, W, B, seed, level) -> residual-bootstrap CI for the SDM rho"
