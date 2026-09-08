"""Sample autocorrelation function."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sample_autocorrelation"]


def _acvf(v, n, max_lag):
    """Autocovariances c_0 .. c_maxlag, divided by n (the biased form)."""
    xbar = sum(v) / n
    d = [t - xbar for t in v]
    return [sum(d[t] * d[t - k] for t in range(k, n)) / n for k in range(max_lag + 1)]


def sample_autocorrelation(y, max_lag=20):
    r"""Sample autocorrelation function of a series.

    .. math::

       r_k = \frac{\sum_{t=k+1}^{n}(y_t-\bar y)(y_{t-k}-\bar y)}
                  {\sum_{t=1}^{n}(y_t-\bar y)^2}

    The denominator is the *full* sum of squares and the numerator has
    only n-k terms, i.e. both divide by n rather than by n-k. That is
    the biased-but-positive-semidefinite convention -- the one R's
    ``acf`` uses, and the one already used by
    :func:`morie.fn.acf.autocorrelation`. It is chosen deliberately:
    dividing by n-k can produce an autocovariance sequence that is not
    a valid covariance function, which then breaks Durbin-Levinson and
    every Yule-Walker fit downstream.

    Parameters
    ----------
    y : array-like
        The series.
    max_lag : int
        Highest lag to return; clipped to n-1.

    Returns
    -------
    RichResult
        Keys ``acf`` (list, index k is lag k, ``acf[0] == 1``),
        ``lags``, ``n``, ``max_lag``, ``ci_bound`` (the +-1.96/sqrt(n)
        white-noise band).

    References
    ----------
    Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis:
    Forecasting and Control*, rev. ed., Holden-Day, sec. 2.1.
    """
    v = [float(t) for t in np.asarray(y, dtype=float).ravel().tolist()]
    n = len(v)
    if n < 3:
        raise ValueError("need at least 3 observations.")
    max_lag = min(int(max_lag), n - 1)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    c = _acvf(v, n, max_lag)
    if c[0] <= 0:
        raise ValueError("series is constant; the autocorrelation is undefined.")
    r = [ck / c[0] for ck in c]
    ci = 1.96 / (n ** 0.5)
    return RichResult(
        payload={
            "acf": r,
            "acvf": c,
            "lags": list(range(max_lag + 1)),
            "n": n,
            "max_lag": max_lag,
            "ci_bound": ci,
            "method": "Sample autocorrelation function (divide-by-n convention)",
        }
    )


def cheatsheet():
    return "acsamp: sample autocorrelation function r_k, k = 0 .. max_lag"
