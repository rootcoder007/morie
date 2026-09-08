# morie.fn -- function file (rootcoder007/morie)
"""Ljung-Box Q test for autocorrelation."""

from __future__ import annotations

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["ljung_box"]


def ljung_box(y, lags=1, fitdf=0):
    """Ljung-Box Q test for residual autocorrelation.

    Formula: ``Q = n(n+2) sum_{k=1}^{m} r_k^2 / (n-k)``, referred to
    ``chi2`` on ``m - fitdf`` degrees of freedom.

    Ljung and Box's point was that Box-Pierce's ``n sum r_k^2`` is badly
    approximated by its chi-square limit in the sample sizes people
    actually have; the ``(n+2)/(n-k)`` factor is a variance correction
    for ``r_k``, whose exact null variance is ``(n-k)/(n(n+2))`` rather
    than ``1/n``.  ``r_k`` uses the biased (positive semi-definite)
    normalisation -- ``n`` in both numerator and denominator -- which is
    what ``stats::acf`` and hence ``stats::Box.test`` use.

    Parameters
    ----------
    y : array-like
        Series, normally a residual series.
    lags : int
        Number of lags ``m`` entering the sum.
    fitdf : int
        Parameters fitted to obtain ``y``; subtracted from the degrees
        of freedom.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``df``, ``acf``, ``n``, ``method``.

    References
    ----------
    Ljung and Box (1978), On a measure of lack of fit in time series
    models, Biometrika 65:297-303.  The paper is paywalled at JSTOR
    (HTTP 403); the statistic was taken instead from R's own
    ``stats::Box.test`` (src/library/stats/R/ts-tests.R, fetched from
    the r-source mirror), which is the canonical reference
    implementation and codes it as
    ``n*(n+2)*sum(1/seq.int(n-1, n-lag)*obs^2)``.
    """
    y = T.vec(y)
    n = len(y)
    m = int(lags)
    fitdf = int(fitdf)
    if m < 1 or n <= m:
        raise ValueError("need 1 <= lags < length(y)")
    r = T.acfbiased(y, m)
    q = 0.0
    for k in range(1, m + 1):
        q += r[k - 1] ** 2 / (n - k)
    q *= n * (n + 2.0)
    df = m - fitdf
    p = 1.0 - stats.chi2.cdf(q, df) if df > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(q),
            "p_value": float(p),
            "df": int(df),
            "acf": r,
            "n": int(n),
            "method": "Ljung-Box Q test",
        }
    )


def cheatsheet():
    return "ljung_box(y, lags, fitdf=0): Ljung-Box Q = n(n+2) sum r_k^2/(n-k)."


# compact alias per ledger/NAMING.md
ljungbox = ljung_box
