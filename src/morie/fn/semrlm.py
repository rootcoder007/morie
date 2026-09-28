# morie.fn -- function file (rootcoder007/morie)
"""Robust LM test for spatial error dependence (Anselin, Bera, Florax and Yoon 1996)."""

from __future__ import annotations

from ._containers import SpatialResult
from .lmtests import lm_spatial_tests


def semrlm(y, X, W) -> SpatialResult:
    """Robust Lagrange multiplier test for a spatial error, robust to a spatial lag.

    ``adjRSerr`` of :func:`~morie.fn.lmtests.lm_spatial_tests` (= ``spdep::lm.RStests``
    ``adjRSerr``), chi-square with 1 df.  It needs ``y`` and ``X``: the
    statistic uses the fitted values as well as the OLS residuals.

    Examples
    --------
    >>> n = 8
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4)]
    >>> round(semrlm([1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1], X, W).statistic, 6)
    0.848354
    """
    r = lm_spatial_tests(y, X, W)["adjRSerr"]
    return SpatialResult(name="semrlm", statistic=r["statistic"], p_value=r["p_value"], extra={"df": 1})


semrlm_fn = semrlm


def cheatsheet() -> str:
    return "semrlm(y, X, W) -> robust LM test for spatial error (spdep adjRSerr)."
