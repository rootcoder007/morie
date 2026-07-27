# morie.fn -- function file (rootcoder007/morie)
"""GARCH(1,1) volatility model."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["garch_model"]


def garch_model(y, p=1, q=1):
    r"""GARCH(1,1) volatility model.

    Tsay Sec. 3.5, p. 131. ``p`` and ``q`` are accepted for interface
    compatibility; the fitted recursion is the (1,1) case, which is
    what the shared core implements.

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    y : array-like
        Return series.
    p : int, default 1
        Order, accepted for interface compatibility.
    q : int, default 1
        Order, accepted for interface compatibility.

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
    fit = garch_fit(y, "garch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "GARCH(1,1) volatility model (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "garchm: GARCH(1,1) volatility model, spec 'garch'"
