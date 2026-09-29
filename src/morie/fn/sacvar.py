# morie.fn -- function file (rootcoder007/morie)
"""SAC variance-covariance matrix."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sacvar(X, W, rho, lam, sigma2, beta):
    r"""SAC variance-covariance matrix.

    Inverse of the analytic information matrix of the SAC model ``y = rho W y + X beta + u``, ``u = lambda W u + e`` at the given
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
    W : array-like, shape (n, n)
        Spatial weights.
    rho, lam : float
        Lag and error parameters.
    sigma2 : float
        Error variance.
    beta : array-like, shape (p,)
        Regression coefficients (enter the ``rho`` blocks).

    Returns
    -------
    SpatialResult
        ``statistic`` is the asymptotic variance of the spatial parameter;
        ``extra`` has ``cov`` and ``information`` (order ``(beta, rho, lambda, sigma2)``; ``statistic`` is ``Var(rho)``), ``se``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood estimators for spatial
    autoregressive models. *Econometrica*, 72(6), 1899-1925.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = sacvar(X, W, 0.3, 0.2, 0.5, [1.0, 2.0])
    >>> round(r.statistic, 12)
    1.663270431041
    """
    info, V = sd.covariance(X, W, beta, float(rho), float(lam), float(sigma2), True, True)
    p = len(sd._mat(X)[0])
    return SpatialResult(
        name="sacvar",
        statistic=V[p][p],
        extra={"cov": V, "information": info, "se": [V[i][i] ** 0.5 for i in range(len(V))]},
    )


sacvar_fn = sacvar


def cheatsheet() -> str:
    return "sacvar(X, W, rho, lam, sigma2, beta) -> asymptotic covariance of (beta, rho, lambda, sigma2)"
