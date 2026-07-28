# morie.fn -- function file (rootcoder007/morie)
"""Average forecast baseline."""

import numpy as np

from ._richresult import RichResult

__all__ = ["average_forecast"]


def average_forecast(y, horizon=1, window=None, seasonal_period=None):
    r"""Forecast every future point as the mean of the history.

    .. math:: \hat y_{T+h} = \frac{1}{T}\sum_{t=1}^{T} y_t

    This is a BASELINE and its value is entirely as a comparison. Any
    forecasting method that cannot beat it on a series is not modelling
    that series, and reporting an impressive-looking MAPE without this
    number is how weak models pass review.

    Under a stationary series the mean forecast is optimal in squared
    error, so beating it requires genuine structure. Under a series
    with a TREND or a unit root it is arbitrarily bad, and the naive
    forecast :math:`\hat y_{T+h} = y_T` is the right baseline instead.
    ``naive_forecast`` and ``drift_forecast`` are returned alongside so
    the comparison is against the appropriate one; ``recommended``
    names it based on the measured autocorrelation of the differences.

    The prediction interval widens with the horizon only through the
    uncertainty in the mean itself, :math:`\sigma\sqrt{1 + 1/T}` -- the
    average forecast does not know that a series can wander, which is
    exactly why it fails on trending data.

    Parameters
    ----------
    y : array-like, shape (T,)
    horizon : int
    window : int, optional
        Use only the last ``window`` observations.
    seasonal_period : int, optional
        Also compute a seasonal-naive baseline.

    Returns
    -------
    RichResult
        ``forecast``, ``naive_forecast``, ``drift_forecast``,
        ``seasonal_naive``, ``ci_lower``, ``ci_upper``,
        ``in_sample_mae``, ``recommended``.

    References
    ----------
    Joseph (2024), *Modern Time Series Forecasting with Python*, 2nd
    ed., chapter 4, baseline forecasts.
    Hyndman and Athanasopoulos (2021), *Forecasting: Principles and
    Practice*, 3rd ed., section 5.2.

    Examples
    --------
    >>> out = average_forecast([1.0, 2.0, 3.0], horizon=2)
    >>> [float(v) for v in out["forecast"]]
    [2.0, 2.0]
    """
    v = np.asarray(y, dtype=float).ravel()
    T = v.size
    if T < 2:
        raise ValueError("need at least 2 observations, got %d." % T)
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1, got %d." % h)
    use = v if window is None else v[-int(window):]
    if use.size < 2:
        raise ValueError("window leaves fewer than 2 observations.")

    mu = float(use.mean())
    sd = float(use.std(ddof=1))
    fc = np.full(h, mu)
    se = sd * np.sqrt(1.0 + 1.0 / use.size)
    z = 1.959963984540054

    naive = np.full(h, float(v[-1]))
    slope = (v[-1] - v[0]) / (T - 1)
    drift = v[-1] + slope * np.arange(1, h + 1)
    seas = None
    if seasonal_period:
        m = int(seasonal_period)
        if m < 1 or m > T:
            raise ValueError(
                "seasonal_period must lie between 1 and the series length."
            )
        seas = np.array([v[-m + ((i) % m)] for i in range(h)])

    diffs = np.diff(v)
    ac1 = (float(np.corrcoef(v[:-1], v[1:])[0, 1]) if T > 2 else np.nan)
    trending = abs(float(np.mean(diffs))) > 0.5 * float(
        np.std(diffs) / np.sqrt(max(T - 1, 1))
    ) * 2.0
    rec = "naive or drift" if (ac1 > 0.9 or trending) else "average"
    return RichResult(
        payload={
            "estimate": fc,
            "forecast": fc,
            "naive_forecast": naive,
            "drift_forecast": drift,
            "seasonal_naive": seas,
            "ci_lower": fc - z * se,
            "ci_upper": fc + z * se,
            "se": float(se),
            "interval_note": (
                "the interval widens only through uncertainty in the mean, "
                "sigma sqrt(1 + 1/T); it does not widen with the horizon, "
                "because this forecast does not know a series can wander"
            ),
            "in_sample_mae": float(np.mean(np.abs(use - mu))),
            "in_sample_rmse": float(np.sqrt(np.mean((use - mu) ** 2))),
            "lag1_autocorrelation": ac1,
            "recommended": rec,
            "recommendation_note": (
                "the mean forecast is optimal under stationarity and "
                "arbitrarily bad under a trend or unit root; lag-1 "
                "autocorrelation of %.2f suggests the %s baseline"
                % (ac1, rec) if ac1 == ac1 else None
            ),
            "baseline_note": (
                "a method that cannot beat this on a series is not modelling "
                "that series; reporting an error metric without a baseline "
                "is how weak models pass review"
            ),
            "n": int(T),
            "n_used": int(use.size),
            "horizon": h,
            "method": "Average (mean) forecast baseline",
        }
    )


def cheatsheet():
    return (
        "joavg: mean forecast baseline with naive and drift alternatives and "
        "a recommendation based on the measured autocorrelation"
    )
