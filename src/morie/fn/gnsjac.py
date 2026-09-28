# morie.fn -- function file (rootcoder007/morie)
"""GNS log-Jacobian log|I - rho W| + log|I - lambda W|."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spdurbin import log_jacobian


def gnsjac(W, rho, lam):
    """GNS log-Jacobian ``log|I - rho W| + log|I - lambda W|`` by LU (:func:`morie.fn.spdurbin.log_jacobian`).

    Examples
    --------
    >>> round(gnsjac([[0, 1], [1, 0]], 0.5, 0.5).statistic, 6)
    -0.575364
    """
    v = log_jacobian(np.asarray(W, dtype=float).tolist(), float(rho), float(lam))
    return SpatialResult(name="gnsjac", statistic=v, p_value=None, extra={})


gnsjac_fn = gnsjac


def cheatsheet() -> str:
    return "gnsjac({}) -> GNS dual Jacobian ln|I-rho*W| + ln|I-lam*W|."
