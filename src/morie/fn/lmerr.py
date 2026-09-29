# morie.fn -- function file (rootcoder007/morie)
"""LM test for spatial error (Anselin 1988)."""

from .lmdiag import _result, _rs_core


def lmerr(y, X, W):
    r"""LM test for spatial error autocorrelation (Burridge 1980; Anselin 1988), RSerr = (u'Wu/sigma2)^2 / T with T = tr((W' + W) W).

    Computed from the OLS fit of y on X (an intercept is prepended
    when X has no constant column); matches spdep::lm.RStests(test =
    "RSerr"). See :func:`morie.fn.lmdiag.lmdiag` for the notation.

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
    >>> r = lmerr([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    3.4904112508
    """
    return _result("lmerr", _rs_core(y, X, W)["RSerr"], 1)


lmerr_fn = lmerr


def cheatsheet() -> str:
    return "lmerr(y, X, W) -> RSerr Rao score test for spatial dependence after OLS (spdep::lm.RStests)."
