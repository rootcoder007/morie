# morie.fn -- function file (rootcoder007/morie)
"""Naive and seasonal-naive forecasts."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["joseph_naive_forecast"]


def joseph_naive_forecast(y, horizon=1, season=None):
    r"""Carry the last value (or last seasonal value) forward.

    .. math::
        \hat y_{T+h} = y_T
        \qquad\text{or}\qquad
        \hat y_{T+h} = y_{T+h-m\lceil h/m\rceil}

    for the seasonal variant with period :math:`m`.

    This is not a toy. The naive forecast is the **benchmark every other
    method must beat**, and for a random walk it is the optimal forecast --
    no model can do better, so a method that merely matches it on such data
    has demonstrated nothing. Scaled error measures like MASE are defined
    against precisely this baseline for that reason.

    Reporting a sophisticated model's accuracy without the naive comparison
    is the most common way a forecasting result looks better than it is.

    Parameters
    ----------
    y : array-like
        Series.
    horizon : int
        Steps ahead, at least 1.
    season : int, optional
        Seasonal period for seasonal-naive. ``None`` gives plain naive.

    Returns
    -------
    RichResult
        ``forecast``, ``method``, ``last_value``, ``in_sample_mae``
        (the MASE denominator).

    References
    ----------
    Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles
        and Practice* (3rd ed.). OTexts.

    Examples
    --------
    Plain naive repeats the last observation.

    >>> import numpy as np
    >>> [float(v) for v in joseph_naive_forecast([1.0, 2.0, 5.0], horizon=3)["forecast"]]
    [5.0, 5.0, 5.0]

    Seasonal naive repeats the last full season, which is what makes it a
    real competitor on seasonal data.

    >>> y = [10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0]
    >>> [float(v) for v in joseph_naive_forecast(y, horizon=4, season=4)["forecast"]]
    [11.0, 21.0, 31.0, 41.0]

    On a random walk the naive forecast is optimal, so beating it is the bar
    any model has to clear.

    >>> rng = np.random.default_rng(0)
    >>> rw = np.cumsum(rng.normal(size=200))
    >>> r = joseph_naive_forecast(rw, horizon=1)
    >>> bool(r["in_sample_mae"] > 0)
    True

    >>> joseph_naive_forecast([1.0, 2.0], horizon=0)
    Traceback (most recent call last):
        ...
    ValueError: horizon must be at least 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if y.size == 0:
        raise ValueError("y must be non-empty")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if season is None:
        fc = np.full(horizon, y[-1])
        name = "naive"
        denom = float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else np.nan
    else:
        m = int(season)
        if m < 1 or m > y.size:
            raise ValueError(f"season must be between 1 and {y.size}")
        fc = np.array([y[-m + ((h - 1) % m)] for h in range(1, horizon + 1)])
        name = f"seasonal naive (m={m})"
        denom = float(np.mean(np.abs(y[m:] - y[:-m]))) if y.size > m else np.nan
    return RichResult(
        title=f"Naive forecast ({name})",
        summary_lines=[("n", int(y.size)), ("horizon", horizon),
                       ("method", name)],
        payload={
            "forecast": fc, "method_used": name, "last_value": float(y[-1]),
            "in_sample_mae": denom, "horizon": horizon, "season": season,
            "method": "joseph_naive_forecast",
        },
    )


def cheatsheet():
    return "jonaiv: the benchmark every model must beat; optimal for a random walk, and the MASE denominator"
