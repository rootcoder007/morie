# morie.fn -- function file (rootcoder007/morie)
"""SAR spillover ratio (indirect/direct)."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .spslx import spatial_impacts


def sarspil(coef, rho, W):
    r"""SAR spillover ratio (indirect/direct).

    The ratio of the average indirect to the average direct impact of each
    covariate in the spatial lag model (LeSage and Pace 2009, sec. 2.7),
    impacts from :func:`morie.fn.spslx.spatial_impacts`. In the lag model it
    equals ``(1'(I - rho W)^{-1}1 - tr((I - rho W)^{-1})) / tr((I - rho
    W)^{-1})`` for every covariate; the share of the total that spills over
    is also returned.

    Parameters
    ----------
    coef : array-like
        Slopes (no intercept).
    rho : float
        Spatial lag parameter.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is the ratio of the first covariate; ``extra`` has
        ``ratio`` and ``share`` (indirect / total) lists and the impacts.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> round(sarspil([2.0], 0.4, [[0, 1, 0], [.5, 0, .5], [0, 1, 0]]).statistic, 12)
    0.478873239437
    """
    r = spatial_impacts(float(rho), sd._vec(coef), sd._mat(W))
    d, i, t = list(r["direct"]), list(r["indirect"]), list(r["total"])
    ratio = [a / b for a, b in zip(i, d)]
    return SpatialResult(
        name="sarspil",
        statistic=ratio[0],
        extra={"ratio": ratio, "share": [a / b for a, b in zip(i, t)], "direct": d, "indirect": i, "total": t},
    )


sarspil_fn = sarspil


def cheatsheet() -> str:
    return "sarspil(coef, rho, W) -> indirect / direct impact ratio"
