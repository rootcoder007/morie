# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the SDEM coefficients [beta, theta] given lambda and sigma^2."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .gnsvar import gnsvar


def sdemvar(X, WX, W, lam, sigma2):
    """SDEM coefficient covariance ``sigma^2 (Z_*' Z_*)^{-1}`` with ``Z = [X, WX]``, ``Z_* = (I - lambda W) Z``.

    ``statistic`` is its trace (see :func:`morie.fn.gnsvar.gnsvar`).

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = sdemvar([[1, 0.0], [1, 1.0], [1, 3.0]], [[0.5], [1.5], [0.5]], W, 0.0, 1.0)
    >>> len(r.extra["cov"])
    3
    """
    Z = [list(a) + list(b) for a, b in zip(np.asarray(X, dtype=float).tolist(), np.asarray(WX, dtype=float).tolist())]
    r = gnsvar(Z, W, 0.0, lam, sigma2)
    return SpatialResult(name="sdemvar", statistic=r.statistic, p_value=None, extra=r.extra)


sdemvar_fn = sdemvar


def cheatsheet() -> str:
    return "sdemvar({}) -> SDEM variance-covariance matrix."
