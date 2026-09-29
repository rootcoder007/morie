# morie.fn -- function file (rootcoder007/morie)
"""SAR direct/indirect/total impact decomposition."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .spslx import spatial_impacts


def sarimp(coef, rho, W):
    r"""SAR direct/indirect/total impact decomposition.

    With ``S_r = (I - rho W)^{-1} beta_r`` the average direct impact is
    ``tr(S_r) / n``, the average total impact ``1'S_r 1 / n`` and the
    indirect (spillover) impact their difference (LeSage and Pace 2009,
    sec. 2.7), exactly ``spatialreg::impacts`` with the exact inverse
    (:func:`morie.fn.spslx.spatial_impacts`). ``coef`` excludes the
    intercept.

    Parameters
    ----------
    coef : array-like
        Slopes of the lag model (no intercept).
    rho : float
        Spatial lag parameter.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is the total impact of the first covariate; ``extra``
        has ``direct``, ``indirect`` and ``total`` lists.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> r = sarimp([2.0], 0.4, [[0, 1, 0], [.5, 0, .5], [0, 1, 0]])
    >>> round(r.extra["direct"][0], 12), round(r.statistic, 12)
    (2.253968253968, 3.333333333333)
    """
    r = spatial_impacts(float(rho), sd._vec(coef), sd._mat(W))
    return SpatialResult(
        name="sarimp",
        statistic=r["total"][0],
        extra={"direct": list(r["direct"]), "indirect": list(r["indirect"]), "total": list(r["total"])},
    )


sarimp_fn = sarimp


def cheatsheet() -> str:
    return "sarimp(coef, rho, W) -> LeSage-Pace direct / indirect / total impacts"
