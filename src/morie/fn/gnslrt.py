# morie.fn -- function file (rootcoder007/morie)
"""GNS likelihood-ratio test against the SDM."""

from __future__ import annotations

from ._containers import SpatialResult
from .spdurbin import spatial_lr_test


def gnslrt(ll_gns, ll_sdm, df=1):
    """GNS likelihood-ratio test against the SDM. ``LR = 2 (l_full - l_restricted)`` against chi-square with ``df`` df.

    Delegates to :func:`morie.fn.spdurbin.spatial_lr_test`.

    Examples
    --------
    >>> r = gnslrt(-10.0, -12.5, 1)
    >>> r.statistic, round(r.p_value, 6)
    (5.0, 0.025347)
    """
    r = spatial_lr_test(ll_gns, ll_sdm, df)
    return SpatialResult(name="gnslrt", statistic=r.statistic, p_value=r.pvalue, extra={"df": r.df})


gnslrt_fn = gnslrt


def cheatsheet() -> str:
    return "gnslrt({}) -> GNS likelihood-ratio test vs SDM."
