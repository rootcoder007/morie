# morie.fn -- function file (rootcoder007/morie)
"""Integrated GARCH fit."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_igarch_fit"]


def vol_igarch_fit(r):
    r"""Integrated GARCH fit.

    Tsay Sec. 3.6, p. 140-141.

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
    Engle, R. F. & Bollerslev, T. (1986). *Econometric Reviews*, 5(1), 1-50.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "igarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Integrated GARCH fit (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volign: Integrated GARCH fit, spec 'igarch'"
