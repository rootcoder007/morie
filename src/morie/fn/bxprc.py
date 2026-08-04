# morie.fn -- function file (rootcoder007/morie)
"""Box-Pierce Q test for autocorrelation."""

from __future__ import annotations

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["box_pierce_test"]


def box_pierce_test(x, lags=1, fitdf=0):
    """Box-Pierce portmanteau test for residual autocorrelation.

    Formula: ``Q = n sum_{k=1}^{m} r_k^2``, referred to ``chi2`` on
    ``m - fitdf`` degrees of freedom.

    This is the uncorrected portmanteau statistic; :func:`ljung_box`
    applies the small-sample variance correction to the same sum and is
    the one to prefer when ``n`` is not large.

    Parameters
    ----------
    x : array-like
        Series, normally a residual series.
    lags : int
        Number of lags ``m`` entering the sum.
    fitdf : int
        Parameters fitted to obtain ``x``.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``df``, ``acf``, ``n``, ``method``.

    References
    ----------
    Box and Pierce (1970), Distribution of residual autocorrelations in
    autoregressive-integrated moving average time series models, JASA
    65:1509-1526.  Paywalled; the statistic was taken from R's
    ``stats::Box.test`` (src/library/stats/R/ts-tests.R, fetched), which
    codes the Box-Pierce branch as ``n*sum(obs^2)``.
    """
    x = T.vec(x)
    n = len(x)
    m = int(lags)
    fitdf = int(fitdf)
    if m < 1 or n <= m:
        raise ValueError("need 1 <= lags < length(x)")
    r = T.acfbiased(x, m)
    q = n * sum(rk * rk for rk in r)
    df = m - fitdf
    p = 1.0 - stats.chi2.cdf(q, df) if df > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(q),
            "p_value": float(p),
            "df": int(df),
            "acf": r,
            "n": int(n),
            "method": "Box-Pierce Q test",
        }
    )


def cheatsheet():
    return "box_pierce_test(x, lags, fitdf=0): Q = n sum r_k^2."


# compact alias per ledger/NAMING.md
boxpiercetest = box_pierce_test
