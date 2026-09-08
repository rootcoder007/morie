# morie.fn -- function file (rootcoder007/morie)
"""Engle-Granger two-step cointegration test."""

from ._coint import engle_granger as _eg
from ._richresult import RichResult

__all__ = ["engle_granger_2step"]


def engle_granger_2step(y, X, lags=1):
    r"""Engle-Granger two-step test for cointegration.

    Step 1 estimates the long-run relation :math:`y_t = \beta'X_t
    + e_t` by OLS; step 2 tests the residual for a unit root. If the
    residual is stationary while the levels are not, the series share
    a common stochastic trend.

    The residual is an *estimated* series, so its ADF statistic is
    compared against MacKinnon (2010) critical values indexed by the
    number of variables -- reusing plain ADF values here would
    over-reject, which is the most common way this test is misapplied.
    Rather than interpolate a false-precision p-value off three
    tabulated points, the result reports a p-value band.

    Parameters
    ----------
    y : array-like, shape (n,)
        Dependent series.
    X : array-like, shape (n,) or (n, k)
        Regressors.
    lags : int, default 1
        Lagged differences in the residual ADF regression.

    Returns
    -------
    RichResult
        keys: ``beta``, ``intercept``, ``residuals``, ``adf_stat``,
        ``p_value_band``, ``critical_values``, ``cointegrated_5pct``,
        ``n_vars``, ``n``, ``method``.

    References
    ----------
    Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error
    correction: representation, estimation, and testing.
    *Econometrica*, 55(2), 251-276.

    MacKinnon, J. G. (2010). Critical values for cointegration tests.
    Queen's Economics Department Working Paper 1227.
    """
    return RichResult(payload=_eg(y, X, lags))


def cheatsheet():
    return "egcoin: OLS long-run relation + ADF on residual, MacKinnon CVs"
