# morie.fn -- function file (rootcoder007/morie)
"""Integrated GARCH(1,1)."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["igarch_integrated"]


def igarch_integrated(x):
    r"""Integrated GARCH(1,1).

    Tsay Sec. 3.6, p. 140-141: the unit root pins alpha at 1 - beta,
    so shocks to variance never die out and persistence is exactly 1.

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    x : array-like
        Return series.

    Returns
    -------
    RichResult
        keys: ``params``, ``sigma2``, ``sigma``, ``loglik``, ``aic``,
        ``bic``, ``persistence``, ``std_residuals``, ``forecast``
        (one-step-ahead variance), ``converged``, ``n``, ``method``.

    References
    ----------
    Engle, R. F. & Bollerslev, T. (1986). Modelling the persistence of
    conditional variances. *Econometric Reviews*, 5(1), 1-50.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(x, "igarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Integrated GARCH(1,1) (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "igarcm: Integrated GARCH(1,1), spec 'igarch'"
