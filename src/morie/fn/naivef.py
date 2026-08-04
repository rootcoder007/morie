"""Naive ŷ_{t+h}=y_t."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["naive_forecast"]


def naive_forecast(y, h=1):
    """
    Naive forecast.

    Formula: yhat_{T+h|T} = y_T for every horizon h

    Verified against Hyndman & Athanasopoulos, *Forecasting: Principles
    and Practice*, 3rd ed., Section 5.2 "Some simple forecasting
    methods" -- source consulted at otexts.com/fpp3, which states
    "For naive forecasts, we simply set all forecasts to be the value
    of the last observation."

    Parameters
    ----------
    y : array-like
        Observed series.
    h : int, optional
        Forecast horizon (default 1).

    Returns
    -------
    RichResult
        Keys: estimate (the h forecasts), last, residual_sd, n, method.
        ``residual_sd`` is the sd of the one-step naive residuals, the
        natural scale for a prediction interval.

    References
    ----------
    Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and
    Practice, 3rd ed. OTexts. Section 5.2.
    """
    v = [float(t) for t in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(v)
    hh = int(h)
    if n < 1:
        raise ValueError("y must be non-empty")
    if hh < 1:
        raise ValueError("h must be at least 1")
    last = v[-1]
    res = [v[i] - v[i - 1] for i in range(1, n)]
    sd = float("nan")
    if len(res) > 1:
        mu = sum(res) / len(res)
        sd = (sum((t - mu) ** 2 for t in res) / (len(res) - 1)) ** 0.5
    return RichResult(
        payload={
            "estimate": [last] * hh,
            "last": last,
            "residual_sd": sd,
            "n": n,
            "method": "Naive forecast yhat_{T+h|T} = y_T -- Hyndman & Athanasopoulos, FPP3 Sec. 5.2",
        }
    )


def cheatsheet():
    return "naivef: Naive ŷ_{t+h}=y_t"
