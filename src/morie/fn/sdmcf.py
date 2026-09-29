# morie.fn -- function file (rootcoder007/morie)
"""SDM common-factor restriction test (Wald)."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmcf(coef, theta, vcov, rho):
    r"""SDM common-factor restriction test (Wald).

    The spatial error model ``y = X beta + (I - lambda W)^{-1} e`` is the
    spatial Durbin model ``y = rho W y + X beta + W X theta + e`` under the
    common-factor restriction ``theta = -rho beta`` (Burridge 1981). The
    Wald test of ``g = theta + rho beta = 0`` uses the delta method: ``G``
    has row ``r`` equal to ``beta_r`` in the ``rho`` column, ``rho`` in the
    ``beta_r`` column and 1 in the ``theta_r`` column, and ``W = g' (G V
    G')^{-1} g`` is chi-square with ``k`` degrees of freedom (Elhorst 2014,
    sec. 2.3; LeSage and Pace 2009, sec. 3.3).

    Parameters
    ----------
    coef : array-like, shape (k,)
        SDM slopes ``beta`` of the lagged regressors (no intercept).
    theta : array-like, shape (k,)
        Coefficients of ``W X``.
    vcov : array-like, shape (2k + 1, 2k + 1)
        Covariance of ``(rho, beta_1..k, theta_1..k)`` in that order.
    rho : float
        SDM spatial lag parameter.

    Returns
    -------
    SpatialResult
        ``statistic``, ``p_value``; ``extra`` has ``df`` and ``g``.

    References
    ----------
    Burridge, P. (1981). Testing for a common factor in a spatial autoregression model.
    *Environment and Planning A*, 13(7), 795-800.

    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*.
    Springer, Berlin.

    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> V = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.01], [0.0, 0.01, 0.09]]
    >>> r = sdmcf([1.0], [-0.2], V, 0.4)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (0.34965034965, 0.5543111254)
    """
    stat, df, p, g = sd.common_factor(coef, theta, vcov, float(rho))
    return SpatialResult(name="sdmcf", statistic=stat, p_value=p, extra={"df": df, "g": g})


sdmcf_fn = sdmcf


def cheatsheet() -> str:
    return "sdmcf(coef, theta, vcov, rho) -> Wald test of theta = -rho beta"
