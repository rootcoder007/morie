# morie.fn -- slice k04 (rootcoder007/morie)
"""Chow (1960) forecast (predictive-failure) test for parameter constancy.

Source: Chow, G. C. (1960).  Tests of equality between sets of
coefficients in two linear regressions.  *Econometrica* 28, 591-605.
The 1960 Econometrica paper is paywalled here and could not be
retrieved; the second of the two tests in that paper -- the forecast
test, which is the one that works when the second sub-sample is shorter
than the number of regressors -- is quoted in its standard published
form::

    F = [ (RSS_c - RSS_1) / n2 ] / [ RSS_1 / (n1 - k) ]  ~  F(n2, n1 - k)

where RSS_1 is the residual sum of squares from fitting the first n1
observations alone, RSS_c that from fitting all n = n1 + n2 together,
and k the number of estimated coefficients (intercept included).

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["chow_forecast_test"]


def _rss(D, y):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ beta
    return float(r @ r)


def chow_forecast_test(y, X, split, add_intercept=True):
    """Chow forecast test at the break point ``split``.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Regressor matrix.
    split : int
        Number ``n1`` of leading observations used for estimation; the
        remaining ``n2 = n - n1`` are the forecast period.
    add_intercept : bool, default True
        Prepend a column of ones to ``X``.

    Returns
    -------
    RichResult
        keys: ``statistic``, ``p_value``, ``df1``, ``df2``, ``rss1``,
        ``rss_pooled``, ``n1``, ``n2``, ``k``, ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n = int(y.size)
    n1 = int(split)
    n2 = n - n1
    D = np.column_stack([np.ones(n), X]) if add_intercept else X
    k = int(D.shape[1])
    if n2 < 1 or n1 - k < 1:
        raise ValueError("need 1 <= n2 and n1 > k")
    rss1 = _rss(D[:n1], y[:n1])
    rssc = _rss(D, y)
    df1, df2 = n2, n1 - k
    stat = ((rssc - rss1) / df1) / (rss1 / df2)
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(stats.f.sf(stat, df1, df2)),
            "df1": int(df1),
            "df2": int(df2),
            "rss1": rss1,
            "rss_pooled": rssc,
            "n1": n1,
            "n2": n2,
            "k": k,
            "method": "Chow (1960) forecast test",
        }
    )


def cheatsheet():
    return "chwfst: Chow forecast test for parameter constancy"
