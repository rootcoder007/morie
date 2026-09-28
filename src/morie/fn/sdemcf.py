# morie.fn -- function file (rootcoder007/morie)
"""Wald test of the SDEM Durbin restriction theta = 0 (SDEM reduces to SEM)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from .spdurbin import spatial_wald_test


def sdemcf(coef, theta, vcov):
    """Wald test of ``theta = 0``, the restriction that reduces the SDEM to the spatial error model.

    ``vcov`` is the covariance of ``theta`` or of ``[coef, theta]`` (its
    trailing ``len(theta)`` block is used). The SDM common-factor hypothesis
    ``theta = -rho beta`` is tested by
    :func:`morie.fn.spdurbin.spatial_lr_test` of SDM against SEM.

    Examples
    --------
    >>> r = sdemcf([0.5, 0.3], [0.1, 0.2], [[0.01, 0.0], [0.0, 0.01]])
    >>> round(r.statistic, 6), r.extra["df"]
    (5.0, 2)
    """
    t = [float(v) for v in np.asarray(theta, dtype=float).tolist()]
    V = np.asarray(vcov, dtype=float).tolist()
    k = len(t)
    if len(V) > k:
        V = [row[-k:] for row in V[-k:]]
    r = spatial_wald_test(t, V)
    return SpatialResult(name="sdemcf", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


sdemcf_fn = sdemcf


def cheatsheet() -> str:
    return "sdemcf({}) -> SDEM common-factor restriction test."
