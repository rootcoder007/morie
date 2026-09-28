# morie.fn -- function file (rootcoder007/morie)
"""Robust LM test for a spatial lag (Anselin, Bera, Florax and Yoon 1996)."""

from __future__ import annotations

from ._containers import SpatialResult
from .lmtests import lm_spatial_tests


def lmrlag2(y, X, W) -> SpatialResult:
    """Robust Lagrange multiplier test for a spatial lag, robust to a spatial error.

    ``adjRSlag`` of :func:`~morie.fn.lmtests.lm_spatial_tests` (= ``spdep::lm.RStests``
    ``adjRSlag``; Bera and Yoon 1993), chi-square with 1 df.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> round(lmrlag2([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W).statistic, 6)
    2.143625
    """
    r = lm_spatial_tests(y, X, W)["adjRSlag"]
    return SpatialResult(name="lmrlag2", statistic=r["statistic"], p_value=r["p_value"], extra={"df": 1})


lmrlag2_fn = lmrlag2


def cheatsheet() -> str:
    return "lmrlag2(y, X, W) -> robust LM test for a spatial lag (spdep adjRSlag)."
