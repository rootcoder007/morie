# morie.fn -- function file (rootcoder007/morie)
"""Gravity model LM test for spatial autocorrelation."""

from __future__ import annotations

from ._containers import SpatialResult
from .semlm import semlm


def igravlm(resid, W):
    r"""Gravity model LM test for spatial autocorrelation.

    The Lagrange multiplier test of spatially autocorrelated errors
    ``LM = (n e'We / e'e)^2 / tr(W'W + WW)`` (Burridge 1980; Anselin 1988)
    applied to the residuals of a log-linear gravity regression, with ``W``
    an origin-destination (network) weights matrix; chi-square(1). This is
    :func:`morie.fn.semlm.semlm`.

    Parameters
    ----------
    resid : array-like, shape (n,)
        OLS residuals of the gravity model.
    W : array-like, shape (n, n)
        Weights between origin-destination pairs.

    Returns
    -------
    SpatialResult
        ``statistic``, ``p_value``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    LeSage, J. P. and Pace, R. K. (2008). Spatial econometric modeling of origin-destination flows.
    *Journal of Regional Science*, 48(5), 941-967.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> round(igravlm([0.3, -0.2, 0.5, -0.4], W).statistic, 12)
    2.395311136052
    """
    r = semlm(resid, W)
    return SpatialResult(name="igravlm", statistic=r.statistic, p_value=r.p_value, extra=r.extra)


igravlm_fn = igravlm


def cheatsheet() -> str:
    return "igravlm(resid, W) -> LM error test on gravity residuals"
