# morie.fn -- function file (rootcoder007/morie)
"""GARCH(1,1) fit."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_garch11_fit"]


def vol_garch11_fit(r):
    r"""GARCH(1,1) fit.

    Tsay Sec. 3.5, p. 131.

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
    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "garch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "GARCH(1,1) fit (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volgar: GARCH(1,1) fit, spec 'garch'"


# compact alias per ledger/NAMING.md
volgarch11fit = vol_garch11_fit
