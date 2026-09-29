# morie.fn -- function file (rootcoder007/morie)
"""SEM common-factor restriction test (Wald)."""

from __future__ import annotations

from ._containers import SpatialResult
from .sdmcf import sdmcf


def semspec(beta, theta, vcov, lam):
    r"""SEM common-factor restriction test (Wald).

    The spatial error model ``y = X beta + (I - lambda W)^{-1} e`` is the
    spatial Durbin model ``y = rho W y + X beta + W X theta + e`` under the
    common-factor restriction ``theta = -rho beta`` (Burridge 1981). The
    Wald test of ``g = theta + rho beta = 0`` uses the delta method: ``G``
    has row ``r`` equal to ``beta_r`` in the ``rho`` column, ``rho`` in the
    ``beta_r`` column and 1 in the ``theta_r`` column, and ``W = g' (G V
    G')^{-1} g`` is chi-square with ``k`` degrees of freedom (Elhorst 2014,
    sec. 2.3; LeSage and Pace 2009, sec. 3.3).
    This is :func:`morie.fn.sdmcf.sdmcf` with the SDM lag parameter written
    ``lam`` (under the null it is the error parameter of the SEM).

    Parameters
    ----------
    beta : array-like, shape (k,)
        SDM slopes of the lagged regressors (no intercept).
    theta : array-like, shape (k,)
        Coefficients of ``W X``.
    vcov : array-like, shape (2k + 1, 2k + 1)
        Covariance of ``(lam, beta, theta)``.
    lam : float
        SDM autoregressive parameter.

    Returns
    -------
    SpatialResult
        ``statistic``, ``p_value``, ``extra["df"]``, ``extra["g"]``.

    References
    ----------
    Burridge, P. (1981). Testing for a common factor in a spatial autoregression model.
    *Environment and Planning A*, 13(7), 795-800.

    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*.
    Springer, Berlin.

    Examples
    --------
    >>> V = [[0.01, 0.0, 0.0], [0.0, 0.04, 0.01], [0.0, 0.01, 0.09]]
    >>> round(semspec([1.0], [-0.2], V, 0.4).statistic, 12)
    0.34965034965
    """
    r = sdmcf(beta, theta, vcov, lam)
    return SpatialResult(name="semspec", statistic=r.statistic, p_value=r.p_value, extra=r.extra)


semspec_fn = semspec


def cheatsheet() -> str:
    return "semspec(beta, theta, vcov, lam) -> common-factor Wald test (SEM vs SDM)"
