# morie.fn -- function file (rootcoder007/morie)
"""GARCH with Student t innovations."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_garch_t"]


def vol_garch_t(r, nu=None):
    r"""GARCH with Student t innovations.

    The t is standardised to unit variance so nu changes the tail
    shape without rescaling the fitted volatility; nu is estimated
    jointly when not supplied.

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    r : array-like
        Return series.
    nu : float, optional
        Shape parameter; estimated jointly when omitted.

    Returns
    -------
    RichResult
        keys: ``params``, ``sigma2``, ``sigma``, ``loglik``, ``aic``,
        ``bic``, ``persistence``, ``std_residuals``, ``forecast``
        (one-step-ahead variance), ``converged``, ``n``, ``method``.

    References
    ----------
    Bollerslev, T. (1987). A conditionally heteroskedastic time series
    model for speculative prices and rates of return. *Review of
    Economics and Statistics*, 69(3), 542-547.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "garch", dist="t", nu=nu)
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "GARCH with Student t innovations (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volgargt: GARCH with Student t innovations, spec 'garch'"
