# morie.fn -- function file (rootcoder007/morie)
"""Threshold (GJR) GARCH."""

from ._garch import garch_fit, garch_forecast
from ._richresult import RichResult

__all__ = ["tgarch_gjr"]


def tgarch_gjr(x, p=1, q=1):
    r"""Threshold (GJR) GARCH.

    Tsay Sec. 3.9, p. 149, eq. (3.34): gamma multiplies the indicator
    of a negative lagged shock, so bad news raises next-period
    variance more than good news of the same size.

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
    Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the
    relation between the expected value and the volatility of the
    nominal excess return on stocks. *Journal of Finance*, 48(5),
    1779-1801.

    Tsay, R. S. (2010). *Analysis of Financial Time Series*
    (3rd ed.). Wiley. Ch. 3 (conditional heteroscedastic models).
    """
    fit = garch_fit(x, "gjr")
    fit["forecast"] = float(garch_forecast(fit, 1)[0])
    fit["method"] = "Threshold (GJR) GARCH (Tsay 2010 Ch. 3)"
    return RichResult(payload=fit)


def cheatsheet():
    return "tgarcm: Threshold (GJR) GARCH, spec 'gjr'"


# compact alias per ledger/NAMING.md
tgarchgjr = tgarch_gjr
