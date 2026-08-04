# morie.fn -- function file (rootcoder007/morie)
"""Drift forecast."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["drift_forecast"]


def drift_forecast(y, h=1):
    r"""Extrapolate the straight line through the first and last observation.

    .. math::
        \hat y_{T+h} = y_T + h\,\frac{y_T - y_1}{T - 1}.

    Equivalent to a random walk with drift, and identical to drawing a line
    between the two endpoints and continuing it -- which is the fact worth
    knowing, because it means the method **ignores everything in between**. A
    series that rises, crashes and recovers to its starting level gets a drift
    of exactly zero, however violent the path.

    The prediction interval widens like :math:`\sqrt{h}` for the random-walk
    component plus a term for drift uncertainty, so it grows faster than for
    the plain naive forecast. That extra width is honest: extrapolating a
    trend is a stronger claim than repeating a level.

    Parameters
    ----------
    y : array-like
        Series, at least 2 points.
    h : int
        Steps ahead.

    Returns
    -------
    RichResult
        ``forecast``, ``drift``, ``se``, ``lower``, ``upper``.

    References
    ----------
    Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles
        and Practice* (3rd ed.). OTexts.

    Examples
    --------
    A clean linear trend is extrapolated exactly.

    >>> import numpy as np
    >>> [float(v) for v in drift_forecast([1.0, 2.0, 3.0, 4.0], h=2)["forecast"]]
    [5.0, 6.0]

    Only the endpoints matter: a series that returns to where it started has
    zero drift regardless of what happened in between.

    >>> float(drift_forecast([10.0, 90.0, 2.0, 10.0], h=1)["drift"])
    0.0

    Intervals widen with the horizon, since extrapolating a trend is a
    stronger claim than repeating a level.

    >>> rng = np.random.default_rng(0)
    >>> r = drift_forecast(np.cumsum(rng.normal(size=100)) + 50, h=10)
    >>> bool(r["se"][-1] > r["se"][0])
    True

    >>> drift_forecast([1.0], h=1)
    Traceback (most recent call last):
        ...
    ValueError: need at least 2 observations to estimate a drift
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = y.size
    if n < 2:
        raise ValueError("need at least 2 observations to estimate a drift")
    h = int(h)
    if h < 1:
        raise ValueError("h must be at least 1")
    drift = float((y[-1] - y[0]) / (n - 1))
    steps = np.arange(1, h + 1)
    fc = y[-1] + steps * drift
    resid = np.diff(y) - drift
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    se = sigma * np.sqrt(steps * (1.0 + steps / max(n - 1, 1)))
    return RichResult(
        title="Drift forecast",
        summary_lines=[("n", int(n)), ("drift", drift), ("h", h)],
        payload={
            "forecast": fc, "drift": drift, "se": se,
            "lower": fc - 1.96 * se, "upper": fc + 1.96 * se,
            "sigma": sigma, "h": h, "method": "drift_forecast",
        },
    )


def cheatsheet():
    return "driftF: line through the FIRST and LAST points only -- everything between is ignored"


# compact alias per ledger/NAMING.md
driftforecast = drift_forecast
