# morie.fn -- function file (rootcoder007/morie)
"""LM test for SARMA (lag + error)."""

from .lmdiag import _result, _rs_core


def lmsarma(y, X, W):
    r"""LM test for SARMA (lag and error jointly), SARMA = adjRSlag + RSerr with 2 df (Anselin, Bera, Florax and Yoon 1996).

    Computed from the OLS fit of y on X (an intercept is prepended
    when X has no constant column); matches spdep::lm.RStests(test =
    "SARMA"). See :func:`morie.fn.lmdiag.lmdiag` for the notation.

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
    >>> r = lmsarma([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    3.7535691455
    """
    return _result("lmsarma", _rs_core(y, X, W)["SARMA"], 2)


lmsarma_fn = lmsarma


def cheatsheet() -> str:
    return "lmsarma(y, X, W) -> SARMA Rao score test for spatial dependence after OLS (spdep::lm.RStests)."
