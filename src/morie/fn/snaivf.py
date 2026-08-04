"""Seasonal naive ŷ_{t+h}=y_{t+h-m}."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["seasonal_naive"]


def seasonal_naive(y, m, h=1):
    """
    Seasonal naive forecast.

    Formula: yhat_{T+h|T} = y_{T+h-m(k+1)}, k = floor((h-1)/m)

    Verified against Hyndman & Athanasopoulos, FPP 3rd ed., Section 5.2
    -- source consulted at otexts.com/fpp3, which gives exactly this
    expression with "k is the integer part of (h-1)/m".

    Parameters
    ----------
    y : array-like
        Observed series.
    m : int
        Seasonal period.
    h : int, optional
        Forecast horizon (default 1).

    Returns
    -------
    RichResult
        Keys: estimate (the h forecasts), m, residual_sd, n, method.

    References
    ----------
    Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and
    Practice, 3rd ed. OTexts. Section 5.2.
    """
    v = [float(t) for t in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(v)
    mm = int(m)
    hh = int(h)
    if mm < 1:
        raise ValueError("m must be at least 1")
    if hh < 1:
        raise ValueError("h must be at least 1")
    if n < mm:
        raise ValueError("y must be at least one full season long")
    fc = []
    for step in range(1, hh + 1):
        k = (step - 1) // mm
        fc.append(v[n + step - mm * (k + 1) - 1])
    res = [v[i] - v[i - mm] for i in range(mm, n)]
    sd = float("nan")
    if len(res) > 1:
        mu = sum(res) / len(res)
        sd = (sum((t - mu) ** 2 for t in res) / (len(res) - 1)) ** 0.5
    return RichResult(
        payload={
            "estimate": fc,
            "m": mm,
            "residual_sd": sd,
            "n": n,
            "method": "Seasonal naive y_{T+h-m(k+1)} -- Hyndman & Athanasopoulos, FPP3 Sec. 5.2",
        }
    )


def cheatsheet():
    return "snaivf: Seasonal naive ŷ_{t+h}=y_{t+h-m}"


# compact alias per ledger/NAMING.md
seasonalnaive = seasonal_naive
