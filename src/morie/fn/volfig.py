# morie.fn -- function file (rootcoder007/morie)
"""Fractionally integrated GARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["vol_figarch_fit"]


def vol_figarch_fit(r, p=1, q=1):
    r"""Fractionally integrated GARCH.

    Between GARCH (d = 0, geometric decay) and IGARCH (d = 1, no
    decay): the ARCH(inf) weights decay hyperbolically.

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    r : array-like
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
    Baillie, R. T., Bollerslev, T. & Mikkelsen, H. O. (1996).
    Fractionally integrated generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(r, "figarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Fractionally integrated GARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "volfig: Fractionally integrated GARCH, spec 'figarch'"
