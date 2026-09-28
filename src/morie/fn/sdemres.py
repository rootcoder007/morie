# morie.fn -- function file (rootcoder007/morie)
"""Moran's I of model residuals (with exact moments when the design is given)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spdurbin import residual_moran


def sdemres(resid, W, X=None):
    """Moran's I ``(n / S0) e'We / e'e`` of residuals; with the design ``X`` the exact ``lm.morantest`` moments.

    Without ``X`` only the statistic is returned; with it
    :func:`morie.fn.spdurbin.residual_moran` gives ``expected``, ``variance``
    and the one-sided p-value.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> round(sdemres([1.0, -1.0, 1.0, -1.0], W).statistic, 6)
    -1.0
    """
    e = [float(v) for v in np.asarray(resid, dtype=float).tolist()]
    Wm = np.asarray(W, dtype=float).tolist()
    if X is not None:
        r = residual_moran(e, np.asarray(X, dtype=float).tolist(), Wm)
        return SpatialResult(
            name="sdemres", statistic=r.I, p_value=r.pvalue, expected=r.expected, variance=r.variance, extra={"z": r.z}
        )
    n = len(e)
    S0 = sum(v for row in Wm for v in row)
    We = [sum(a * b for a, b in zip(row, e)) for row in Wm]
    stat = n / S0 * sum(a * b for a, b in zip(e, We)) / sum(a * a for a in e)
    return SpatialResult(name="sdemres", statistic=stat, p_value=None, extra={})


sdemres_fn = sdemres


def cheatsheet() -> str:
    return "sdemres({}) -> SDEM residual autocorrelation check."
