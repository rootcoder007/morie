# morie.fn -- function file (rootcoder007/morie)
"""EGARCH fit."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_egarch_fit"]


def vol_egarch_fit(r):
    r"""EGARCH fit.

    Tsay Sec. 3.8, p. 143.

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
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset
    returns: a new approach. *Econometrica*, 59(2), 347-370.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "egarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "EGARCH fit (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volegar: EGARCH fit, spec 'egarch'"


# compact alias per ledger/NAMING.md
volegarchfit = vol_egarch_fit
