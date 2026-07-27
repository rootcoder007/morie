# morie.fn -- function file (rootcoder007/morie)
"""GARCH with GED innovations."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_garch_ged"]


def vol_garch_ged(r, nu=None):
    r"""GARCH with GED innovations.

    nu = 2 recovers the Gaussian; nu < 2 is fat-tailed, nu > 2 thin.

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
    Nelson, D. B. (1991). *Econometrica*, 59(2), 347-370 (Sec. 4,
    the generalized error distribution).

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "garch", dist="ged", nu=nu)
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "GARCH with GED innovations (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volgargd: GARCH with GED innovations, spec 'garch'"
