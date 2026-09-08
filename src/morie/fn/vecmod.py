# morie.fn -- function file (rootcoder007/morie)
"""Vector error-correction model."""

from ._coint import vecm_fit
from ._richresult import RichResult

__all__ = ["vector_error_correction"]


def vector_error_correction(X, r=1, lags=1):
    r"""VECM at a fixed cointegrating rank.

    .. math:: \Delta Y_t = \alpha\beta'Y_{t-1}
              + \sum_{i=1}^{p}\Gamma_i \Delta Y_{t-i} + \epsilon_t

    beta (the long-run relations) comes from the Johansen
    eigenvectors; alpha and the short-run Gamma matrices are OLS given
    beta. The Johansen-selected rank is returned alongside the rank
    you asked for, so a mismatch between the two is visible rather
    than silent.

    alpha is the substantive output: a row near zero means that series
    does not adjust to disequilibrium -- it is weakly exogenous, and
    the others do the correcting.

    Parameters
    ----------
    X : array-like, shape (T, n)
        Level series, 2 <= n <= 6.
    r : int, default 1
        Cointegrating rank.
    lags : int, default 1
        Lagged differences.

    Returns
    -------
    RichResult
        keys: ``alpha`` (n, r), ``beta`` (n, r), ``gamma`` (list),
        ``intercept``, ``ect``, ``residuals``, ``sigma``,
        ``eigenvalues``, ``johansen_rank_5pct``, ``rank``, ``lags``,
        ``T``, ``method``.

    References
    ----------
    Johansen, S. (1991). Estimation and hypothesis testing of
    cointegration vectors in Gaussian vector autoregressive models.
    *Econometrica*, 59(6), 1551-1580.

    Hamilton, J. D. (1994). *Time Series Analysis*. Princeton
    University Press. Ch. 19-20.
    """
    return RichResult(payload=vecm_fit(X, r, lags))


def cheatsheet():
    return "vecmod: dY = alpha beta' Y_{t-1} + Gamma dY + e; alpha = adjustment speeds"
