# morie.fn -- function file (rootcoder007/morie)
"""Robust LM test for spatial lag."""

from .lmdiag import _result, _rs_core


def lmrlag(y, X, W):
    r"""Robust LM test for a spatial lag (Anselin, Bera, Florax and Yoon 1996), adjRSlag = (d_lag - d_err)^2 / (nJ - T), robust to local spatial error.

    Computed from the OLS fit of y on X (an intercept is prepended
    when X has no constant column); matches spdep::lm.RStests(test =
    "adjRSlag"). See :func:`morie.fn.lmdiag.lmdiag` for the notation.

    References
    ----------
    Anselin, L. (1988). Lagrange multiplier test diagnostics for spatial
    dependence and spatial heterogeneity. *Geographical Analysis* 20, 1-17.
    Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple
    diagnostic tests for spatial dependence. *Regional Science and Urban
    Economics* 26, 77-104.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmrlag([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    0.2631578947
    """
    return _result("lmrlag", _rs_core(y, X, W)["adjRSlag"], 1)


lmrlag_fn = lmrlag


def cheatsheet() -> str:
    return "lmrlag(y, X, W) -> adjRSlag Rao score test for spatial dependence after OLS (spdep::lm.RStests)."
