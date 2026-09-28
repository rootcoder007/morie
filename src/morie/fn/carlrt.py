# morie.fn -- function file (rootcoder007/morie)
"""CAR likelihood-ratio test against the non-spatial model."""

from __future__ import annotations

from ._containers import SpatialResult
from .spdurbin import spatial_lr_test


def carlrt(ll_car, ll_null, df=1):
    """CAR likelihood-ratio test against the non-spatial model. ``LR = 2 (l_full - l_restricted)`` against chi-square with ``df`` df.

    Delegates to :func:`morie.fn.spdurbin.spatial_lr_test`.

    Examples
    --------
    >>> r = carlrt(-10.0, -12.5, 1)
    >>> r.statistic, round(r.p_value, 6)
    (5.0, 0.025347)
    """
    r = spatial_lr_test(ll_car, ll_null, df)
    return SpatialResult(name="carlrt", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


carlrt_fn = carlrt


def cheatsheet() -> str:
    return "carlrt({}) -> CAR likelihood-ratio test."
