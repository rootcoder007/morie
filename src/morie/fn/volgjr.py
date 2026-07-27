# morie.fn -- function file (rootcoder007/morie)
"""GJR-GARCH fit."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_gjr_garch"]


def vol_gjr_garch(r):
    r"""GJR-GARCH fit.

    Tsay Sec. 3.9, p. 149, eq. (3.34).

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
    Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993).
    *Journal of Finance*, 48(5), 1779-1801.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "gjr")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "GJR-GARCH fit (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volgjr: GJR-GARCH fit, spec 'gjr'"
