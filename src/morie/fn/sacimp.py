# morie.fn -- function file (rootcoder007/morie)
"""SAC direct/indirect/total impacts."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .spslx import spatial_impacts


def sacimp(coef, rho, lam, W):
    r"""SAC direct/indirect/total impacts.

    With ``S_r = (I - rho W)^{-1} beta_r`` the average direct impact is
    ``tr(S_r) / n``, the average total impact ``1'S_r 1 / n`` and the
    indirect (spillover) impact their difference (LeSage and Pace 2009,
    sec. 2.7), exactly ``spatialreg::impacts`` with the exact inverse
    (:func:`morie.fn.spslx.spatial_impacts`). ``coef`` excludes the
    intercept.
    The error parameter ``lambda`` does not enter: ``E(y | X) = (I - rho
    W)^{-1} X beta`` in the SAC model, so its impacts are those of the lag
    part (LeSage and Pace 2009, sec. 3.1); ``lam`` is kept for the
    signature and reported back.

    Parameters
    ----------
    coef : array-like
        Slopes (no intercept).
    rho : float
        Spatial lag parameter.
    lam : float
        Spatial error parameter (does not affect the impacts).
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is the total impact of the first covariate; ``extra``
        has ``direct``, ``indirect``, ``total`` and ``lambda``.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> r = sacimp([2.0], 0.4, 0.3, [[0, 1, 0], [.5, 0, .5], [0, 1, 0]])
    >>> round(r.extra["indirect"][0], 12)
    1.079365079365
    """
    r = spatial_impacts(float(rho), sd._vec(coef), sd._mat(W))
    return SpatialResult(
        name="sacimp",
        statistic=r["total"][0],
        extra={
            "direct": list(r["direct"]),
            "indirect": list(r["indirect"]),
            "total": list(r["total"]),
            "lambda": float(lam),
        },
    )


sacimp_fn = sacimp


def cheatsheet() -> str:
    return "sacimp(coef, rho, lam, W) -> impacts of the SAC lag part"
