# morie.fn -- function file (rootcoder007/morie)
"""Threshold GARCH fit."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_tgarch_fit"]


def vol_tgarch_fit(r):
    r"""Threshold GARCH fit.

    Tsay Sec. 3.9, p. 149. Zakoian's original form models sigma
    rather than sigma^2; this fit uses the eq. (3.34) variance form
    the chapter states, and the docstring says so rather than
    implying the level form.

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
    Zakoian, J.-M. (1994). Threshold heteroskedastic models. *Journal
    of Economic Dynamics and Control*, 18(5), 931-955.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "tgarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Threshold GARCH fit (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "voltgr: Threshold GARCH fit, spec 'tgarch'"
