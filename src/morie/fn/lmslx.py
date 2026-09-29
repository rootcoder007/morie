# morie.fn -- function file (rootcoder007/morie)
"""LM test for SLX (spatially lagged X)."""

from .lmdiag import _result, _rs_core


def lmslx(y, X, W):
    r"""Rao score test for spatially lagged regressors, gamma = 0 in y = X beta + W X gamma + e.

    With WX the lags of the non-constant regressors and M = I -
    X(X'X)^{-1}X', RS_WX = u'WX (X'W'MWX)^{-1} X'W'u / sigma2 with
    k_x degrees of freedom (Koley and Bera 2024; spreg.lm_wx,
    spdep::SD.RStests(test = "SDM_RSWX")). extra also carries the
    SLX-plus-error test RSerr + RS_WX (k_x + 1 df).

    References
    ----------
    Koley, M. and Bera, A. K. (2024). To use, or not to use the spatial
    Durbin model? That is the question. *Spatial Economic Analysis* 19,
    30-56.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = lmslx([1.0, 2.5, 2.0, 4.5, 4.0], [[0.0], [1.0], [2.0], [3.0], [4.0]], W)
    >>> round(r.statistic, 10)
    0.2631578947
    """
    c = _rs_core(y, X, W)
    if not c["kx"]:
        raise ValueError("X needs at least one non-constant column")
    return _result("lmslx", c["RSWX"], c["kx"], {"RSerr_WX": c["RSerr_WX"]})


lmslx_fn = lmslx


def cheatsheet() -> str:
    return "lmslx(y, X, W) -> Rao score test for WX (gamma = 0), Koley-Bera (2024)."
