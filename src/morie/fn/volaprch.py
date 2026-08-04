# morie.fn -- function file (rootcoder007/morie)
"""Asymmetric power ARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_aparch_fit"]


def vol_aparch_fit(r):
    r"""Asymmetric power ARCH.

    Nests GARCH (delta = 2, gamma = 0) and the Taylor/Schwert model
    (delta = 1); the power delta is estimated rather than fixed.

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
    Ding, Z., Granger, C. W. J. & Engle, R. F. (1993). A long memory
    property of stock market returns and a new model. *Journal of
    Empirical Finance*, 1(1), 83-106.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "aparch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Asymmetric power ARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volaprch: Asymmetric power ARCH, spec 'aparch'"


# compact alias per ledger/NAMING.md
volaparchfit = vol_aparch_fit
