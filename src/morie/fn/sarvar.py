# morie.fn -- function file (rootcoder007/morie)
"""SAR variance-covariance of estimator."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sarvar(X, W, rho, sigma2, beta):
    r"""SAR variance-covariance of estimator.

    Inverse of the analytic information matrix of the spatial lag model ``y = rho W y + X beta + e`` at the given
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
    rho : float
        Spatial lag parameter.
    sigma2 : float
        Error variance.
    beta : array-like, shape (p,)
        Regression coefficients (enter the ``rho`` blocks).

    Returns
    -------
    SpatialResult
        ``statistic`` is the asymptotic variance of the spatial parameter;
        ``extra`` has ``cov`` and ``information`` (order ``(beta, rho, sigma2)``), ``se``.

    References
    ----------
    Ord, K. (1975). Estimation methods for models of spatial interaction. *Journal of the American
    Statistical Association*, 70(349), 120-126.

    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood estimators for spatial
    autoregressive models. *Econometrica*, 72(6), 1899-1925.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = sarvar(X, W, 0.3, 0.5, [1.0, 2.0])
    >>> round(r.statistic, 12)
    0.103499213318
    """
    info, V = sd.covariance(X, W, beta, float(rho), 0.0, float(sigma2), True, False)
    p = len(sd._mat(X)[0])
    return SpatialResult(
        name="sarvar",
        statistic=V[p][p],
        extra={"cov": V, "information": info, "se": [V[i][i] ** 0.5 for i in range(len(V))]},
    )


sarvar_fn = sarvar


def cheatsheet() -> str:
    return "sarvar(X, W, rho, sigma2, beta) -> asymptotic covariance of (beta, rho, sigma2)"
