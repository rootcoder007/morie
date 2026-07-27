# morie.fn -- function file (rootcoder007/morie)
"""Nelson's EGARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["egarch_nelson"]


def egarch_nelson(x, p=1, q=1):
    r"""Nelson's EGARCH.

    Tsay Sec. 3.8, p. 143, eq. (3.24)-(3.25).

    Fitted by Gaussian quasi-maximum-likelihood on the shared
    recursion in :mod:`morie.fn._garch`.

    Parameters
    ----------
    x : array-like
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
    Nelson, D. B. (1991). *Econometrica*, 59(2), 347-370.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(x, "egarch")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Nelson's EGARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "egarcm: Nelson's EGARCH, spec 'egarch'"
