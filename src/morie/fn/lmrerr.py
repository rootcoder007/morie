# morie.fn -- function file (rootcoder007/morie)
"""Robust LM test for spatial error."""

from .lmdiag import _result, _rs_core


def lmrerr(y, X, W):
    r"""Robust LM test for spatial error (Anselin, Bera, Florax and Yoon 1996), adjRSerr = (d_err - T d_lag/nJ)^2 / (T(1 - T/nJ)), robust to a local spatial lag.

    Computed from the OLS fit of y on X (an intercept is prepended
    when X has no constant column); matches spdep::lm.RStests(test =
    "adjRSerr"). See :func:`morie.fn.lmdiag.lmdiag` for the notation.

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
    >>> r = lmrerr([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    1.1253689679
    """
    return _result("lmrerr", _rs_core(y, X, W)["adjRSerr"], 1)


lmrerr_fn = lmrerr


def cheatsheet() -> str:
    return "lmrerr(y, X, W) -> adjRSerr Rao score test for spatial dependence after OLS (spdep::lm.RStests)."
