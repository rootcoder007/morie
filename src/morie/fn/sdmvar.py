# morie.fn -- function file (rootcoder007/morie)
"""SDM variance-covariance matrix."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmvar(X, WX, W, rho, sigma2, beta):
    r"""SDM variance-covariance matrix.

    Inverse of the analytic information matrix of the spatial Durbin model, the lag model on ``Z = [X, W X_*]`` at the given
    parameters (Ord 1975; Anselin 1988, ch. 6; Lee 2004). With ``A = I -
    rho W``, ``B = I - lambda W``, ``W_A = W A^{-1}``, ``W_B = W B^{-1}``,
    ``C = B W_A B^{-1}`` and ``g = B W_A X beta``: ``I_bb = (BX)'BX /
    sigma^2``, ``I_b,rho = (BX)'g / sigma^2``, ``I_rho,rho = tr(CC) +
    tr(C'C) + g'g / sigma^2``, ``I_rho,lam = tr(W_B C) + tr(W_B' C)``,
    ``I_lam,lam = tr(W_B W_B) + tr(W_B' W_B)``, ``I_rho,s2 = tr(W_A) /
    sigma^2``, ``I_lam,s2 = tr(W_B) / sigma^2``, ``I_s2,s2 = n / (2
    sigma^4)``, the other blocks zero (terms of an absent parameter
    dropped). This is the ``asy`` covariance ``spatialreg`` reports for
    ``lagsarlm`` / ``errorsarlm`` with ``method = "eigen"``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix including any intercept column.
    WX : array-like, shape (n, q)
        Lagged regressors ``W X_*``.
    W : array-like, shape (n, n)
        Spatial weights.
    rho : float
        Spatial lag parameter.
    sigma2 : float
        Error variance.
    beta : array-like, shape (p + q,)
        Coefficients of ``[X, WX]``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the asymptotic variance of the spatial parameter;
        ``extra`` has ``cov`` and ``information`` (order ``(beta, theta, rho, sigma2)``), ``se``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = sdmvar(X, [[sum(w * r[1] for w, r in zip(row, X))] for row in W], W, 0.3, 0.5, [1.0, 2.0, 0.5])
    >>> round(r.statistic, 12)
    0.111063457737
    """
    Z = [list(a) + list(b) for a, b in zip(sd._mat(X), sd._mat(WX))]
    info, V = sd.covariance(Z, W, beta, float(rho), 0.0, float(sigma2), True, False)
    p = len(Z[0])
    return SpatialResult(
        name="sdmvar",
        statistic=V[p][p],
        extra={"cov": V, "information": info, "se": [V[i][i] ** 0.5 for i in range(len(V))]},
    )


sdmvar_fn = sdmvar


def cheatsheet() -> str:
    return "sdmvar(X, WX, W, rho, sigma2, beta) -> asymptotic covariance of (beta, theta, rho, sigma2)"
