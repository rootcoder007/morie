# morie.fn -- function file (rootcoder007/morie)
"""Exponential GARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["egarch_model"]


def egarch_model(y, p=1, q=1):
    r"""Exponential GARCH.

    Tsay Sec. 3.8, p. 143, eq. (3.24)-(3.25). Modelling log variance
    means omega may be negative and sigma^2 stays positive without a
    constraint; E|z| = sqrt(2/pi) for a Gaussian (Remark, p. 143).

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
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset
    returns: a new approach. *Econometrica*, 59(2), 347-370.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(y, "egarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Exponential GARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "egarch: Exponential GARCH, spec 'egarch'"
