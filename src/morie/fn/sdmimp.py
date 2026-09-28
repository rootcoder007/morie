# morie.fn -- function file (rootcoder007/morie)
"""Direct, indirect and total impacts of a spatial Durbin model."""

from __future__ import annotations

from ._containers import SpatialResult
from .spslx import spatial_impacts


def sdmimp(coef, theta, rho, W) -> SpatialResult:
    """SDM impacts ``S_r = (I - rho W)^{-1}(beta_r I + theta_r W)`` (LeSage and Pace 2009).

    Delegates to :func:`~morie.fn.spslx.spatial_impacts` (= ``spatialreg::impacts``);
    ``coef`` and ``theta`` exclude the intercept.  ``statistic`` is the total
    impact of the first covariate.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = sdmimp([2.0], [0.5], 0.4, W)
    >>> round(r.statistic, 6), round(r.extra["direct"][0], 6)
    (4.166667, 2.412698)
    """
    r = spatial_impacts(rho, coef, W, theta=theta)
    return SpatialResult(
        name="sdmimp",
        statistic=r["total"][0],
        p_value=None,
        extra={"direct": r["direct"], "indirect": r["indirect"], "total": r["total"]},
    )


sdmimp_fn = sdmimp


def cheatsheet() -> str:
    return "sdmimp(coef, theta, rho, W) -> SDM direct/indirect/total impacts."
