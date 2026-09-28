# morie.fn -- function file (rootcoder007/morie)
"""GNS direct, indirect and total impacts (LeSage and Pace 2009)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spslx import spatial_impacts


def gnsimp(coef, theta, rho, lam, W):
    """GNS impacts ``S_r = (I - rho W)^{-1} (beta_r I + theta_r W)``; ``statistic`` is the first total impact.

    The error parameter ``lam`` does not enter the impacts. Delegates to
    :func:`morie.fn.spslx.spatial_impacts`.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> round(gnsimp([2.0], [0.0], 0.4, 0.1, W).statistic, 6)
    3.333333
    """
    r = spatial_impacts(float(rho), coef, np.asarray(W, dtype=float).tolist(), theta)
    return SpatialResult(
        name="gnsimp",
        statistic=float(r["total"][0]),
        p_value=None,
        extra={k: list(r[k]) for k in ("direct", "indirect", "total")},
    )


gnsimp_fn = gnsimp


def cheatsheet() -> str:
    return "gnsimp({}) -> GNS direct/indirect/total impacts."
