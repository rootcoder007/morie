# morie.fn -- function file (rootcoder007/morie)
"""SDEM log-Jacobian log|I - lambda W|."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spdurbin import log_jacobian


def sdemjac(W, lam):
    """SDEM log-Jacobian ``log|I - lambda W|`` by LU (:func:`morie.fn.spdurbin.log_jacobian`).

    Examples
    --------
    >>> round(sdemjac([[0, 1], [1, 0]], 0.5).statistic, 6)
    -0.287682
    """
    v = log_jacobian(np.asarray(W, dtype=float).tolist(), float(lam))
    return SpatialResult(name="sdemjac", statistic=v, p_value=None, extra={})


sdemjac_fn = sdemjac


def cheatsheet() -> str:
    return "sdemjac({}) -> SDEM Jacobian term."
