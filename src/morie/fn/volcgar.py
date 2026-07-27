# morie.fn -- function file (rootcoder007/morie)
"""Component GARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_cgarch_fit"]


def vol_cgarch_fit(r):
    r"""Component GARCH.

    Splits volatility into a slow permanent component q_t and a fast
    transitory deviation, so a single persistence number cannot
    describe the series -- the fit returns both.

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    r : array-like
        Return series.

    Returns
    -------
    RichResult
        keys: ``params``, ``sigma2``, ``sigma``, ``loglik``, ``aic``,
        ``bic``, ``persistence``, ``std_residuals``, ``forecast``
        (one-step-ahead variance), ``converged``, ``n``, ``method``.

    References
    ----------
    Engle, R. F. & Lee, G. G. J. (1999). A permanent and transitory
    component model of stock return volatility. In *Cointegration,
    Causality and Forecasting*, Oxford University Press, 475-497.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "cgarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Component GARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volcgar: Component GARCH, spec 'cgarch'"
