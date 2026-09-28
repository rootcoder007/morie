# morie.fn -- function file (rootcoder007/morie)
"""Wald test that the GNS spatial parameters are jointly zero."""

from __future__ import annotations

from ._containers import SpatialResult
from .spdurbin import spatial_wald_test


def gnswald(params, vcov):
    """Wald test ``b' V^{-1} b`` of ``(rho, lambda) = 0`` against chi-square (:func:`morie.fn.spdurbin.spatial_wald_test`).

    Examples
    --------
    >>> r = gnswald([0.3, 0.2], [[0.01, 0.0], [0.0, 0.01]])
    >>> round(r.statistic, 6), r.extra["df"]
    (13.0, 2)
    """
    r = spatial_wald_test(params, vcov)
    return SpatialResult(name="gnswald", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


gnswald_fn = gnswald


def cheatsheet() -> str:
    return "gnswald({}) -> GNS Wald test on rho and lambda."
