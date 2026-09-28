# morie.fn -- function file (rootcoder007/morie)
"""SDEM direct, indirect and total impacts (Halleck Vega and Elhorst 2015)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult


def sdemimp(coef, theta, lam, W):
    """SDEM impacts: direct ``beta_r``, indirect ``theta_r``, total ``beta_r + theta_r`` (``lam`` and ``W`` do not enter).

    ``statistic`` is the first total impact; use
    :func:`morie.fn.spdurbin.sdem_ml` for impacts with standard errors.

    Examples
    --------
    >>> sdemimp([0.5, 0.3], [0.1, 0.2], 0.3, [[0, 1], [1, 0]]).extra["total"]
    [0.6, 0.5]
    """
    b = [float(v) for v in np.asarray(coef, dtype=float).tolist()]
    t = [float(v) for v in np.asarray(theta, dtype=float).tolist()]
    if len(b) != len(t):
        raise ValueError("theta must match coef")
    tot = [u + v for u, v in zip(b, t)]
    return SpatialResult(
        name="sdemimp", statistic=tot[0], p_value=None, extra={"direct": b, "indirect": t, "total": tot}
    )


sdemimp_fn = sdemimp


def cheatsheet() -> str:
    return "sdemimp({}) -> SDEM direct/indirect/total impacts."
