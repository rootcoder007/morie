# morie.fn -- function file (rootcoder007/morie)
"""SDEM likelihood-ratio test against the SEM."""

from __future__ import annotations

from ._containers import SpatialResult
from .spdurbin import spatial_lr_test


def sdemlrt(ll_sdem, ll_sem, df=2):
    """SDEM likelihood-ratio test against the SEM. ``LR = 2 (l_full - l_restricted)`` against chi-square with ``df`` df.

    Delegates to :func:`morie.fn.spdurbin.spatial_lr_test`.

    Examples
    --------
    >>> r = sdemlrt(-10.0, -12.5, 1)
    >>> r.statistic, round(r.p_value, 6)
    (5.0, 0.025347)
    """
    r = spatial_lr_test(ll_sdem, ll_sem, df)
    return SpatialResult(name="sdemlrt", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


sdemlrt_fn = sdemlrt


def cheatsheet() -> str:
    return "sdemlrt({}) -> SDEM likelihood-ratio test vs SEM."
