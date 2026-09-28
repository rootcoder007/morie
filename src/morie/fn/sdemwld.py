# morie.fn -- function file (rootcoder007/morie)
"""Wald test of the SDEM error parameter lambda."""

from __future__ import annotations

from ._containers import SpatialResult
from .spdurbin import spatial_wald_test


def sdemwld(lam, se_lam):
    """Wald test ``(lambda / se)^2`` of ``lambda = 0`` against chi-square with 1 df.

    Examples
    --------
    >>> r = sdemwld(0.3, 0.1)
    >>> round(r.statistic, 6), round(r.p_value, 6)
    (9.0, 0.0027)
    """
    r = spatial_wald_test([float(lam)], [[float(se_lam) ** 2]])
    return SpatialResult(name="sdemwld", statistic=r.statistic, p_value=r.pvalue, extra={"df": 1})


sdemwld_fn = sdemwld


def cheatsheet() -> str:
    return "sdemwld({}) -> SDEM Wald test on lambda."
