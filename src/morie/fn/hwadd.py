# morie.fn -- function file (rootcoder007/morie)
"""Holt-Winters additive seasonal method."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["holt_winters_additive"]


def holt_winters_additive(y, period=4, alpha=0.3, beta=0.1, gamma=0.1, horizon=None):
    r"""Triple exponential smoothing with additive seasonality.

    Level, trend and seasonal components each get their own smoothing
    parameter:

    .. math::
        \ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
        b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1} \\
        s_t &= \gamma\,(y_t - \\ell_t) + (1-\gamma)s_{t-m}

    with forecast :math:`\hat y_{T+h} = \\ell_T + h b_T + s_{T+h-m}`.

    Additive seasonality means the seasonal swing is a **fixed number of
    units** regardless of level -- December is 500 units above trend whether
    the trend is at 1000 or 10000. When the swing instead grows with the
    level, this systematically under-forecasts the peaks of a growing series
    and over-forecasts its troughs; the multiplicative form exists for that
    case.

    Additive seasonality is the safe default when the series can be zero or
    negative, where the multiplicative form is undefined.

    Three parameters on a short series is a lot to estimate: at least two full
    seasons are required to initialise the seasonal indices at all, and
    several more before they mean anything. The initialisation used here is
    the classical decomposition of the first two periods.

    Parameters
    ----------
    y : array-like
        Series, at least ``2 * period`` observations.
    period : int
        Seasonal period.
    alpha, beta, gamma : float
        Smoothing parameters for level, trend and season, each in [0, 1].
    horizon : int, optional
        Steps ahead. Defaults to one full season.

    Returns
    -------
    RichResult
        ``forecast``, ``level``, ``trend``, ``seasonal``, ``fitted``,
        ``residuals``, ``sse``.

    References
    ----------
    Winters, P. R. (1960). Forecasting sales by exponentially weighted moving
        averages. *Management Science*, 6(3), 324-342.
    Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles
        and Practice* (3rd ed.). OTexts.

    Examples
    --------
    Recovers a level, trend and seasonal pattern from a series built with
    all three.

    >>> import numpy as np
    >>> t = np.arange(48)
    >>> y = 100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12)
    >>> r = holt_winters_additive(y, period=12, alpha=0.4, beta=0.1, gamma=0.3)
    >>> bool(r["trend"] > 1.0)
    True

    The forecast keeps the seasonal shape rather than flattening it.

    >>> bool(np.ptp(r["forecast"]) > 10)
    True

    Additive seasonality tolerates zero and negative values, where the
    multiplicative form cannot go.

    >>> z = np.r_[np.zeros(12), -5 * np.ones(12), np.zeros(12), np.ones(12)]
    >>> bool(np.all(np.isfinite(holt_winters_additive(z, period=12)["forecast"])))
    True

    >>> holt_winters_additive([1.0] * 5, period=12)
    Traceback (most recent call last):
        ...
    ValueError: need at least 24 observations for period 12
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    m = int(period)
    if m < 2:
        raise ValueError("period must be at least 2")
    if y.size < 2 * m:
        raise ValueError(f"need at least {2 * m} observations for period {m}")
    for nm, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{nm} must be in [0, 1]")
    horizon = m if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    MULT = 'additive' == "multiplicative"
    if MULT and np.any(y <= 0):
        raise ValueError("multiplicative seasonality needs strictly positive data")

    # Classical initialisation from the first two full seasons.
    s0 = y[:m].mean()
    s1 = y[m:2 * m].mean()
    level = s0
    trend = (s1 - s0) / m
    season = (y[:m] / s0) if MULT else (y[:m] - s0)

    fitted = np.empty(y.size)
    seas = list(season)
    for t in range(y.size):
        sea = seas[t]
        fitted[t] = (level + trend) * sea if MULT else level + trend + sea
        prev = level
        if MULT:
            level = alpha * (y[t] / max(sea, 1e-12)) + (1 - alpha) * (level + trend)
            new_sea = gamma * (y[t] / max(level, 1e-12)) + (1 - gamma) * sea
        else:
            level = alpha * (y[t] - sea) + (1 - alpha) * (level + trend)
            new_sea = gamma * (y[t] - level) + (1 - gamma) * sea
        trend = beta * (level - prev) + (1 - beta) * trend
        seas.append(new_sea)

    steps = np.arange(1, horizon + 1)
    tail = np.array([seas[-m + ((h - 1) % m)] for h in steps])
    fc = (level + steps * trend) * tail if MULT else level + steps * trend + tail
    resid = y - fitted
    return RichResult(
        title="Holt-Winters (additive)",
        summary_lines=[("n", int(y.size)), ("period", m),
                       ("level", float(level)), ("trend", float(trend))],
        payload={
            "forecast": fc, "level": float(level), "trend": float(trend),
            "seasonal": np.array(seas[-m:]), "fitted": fitted,
            "residuals": resid, "sse": float(np.sum(resid**2)),
            "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "period": m, "horizon": horizon, "method": "holt_winters_additive",
        },
    )


def cheatsheet():
    return "hwadd: additive season; needs 2+ full periods to initialise, 3 params is a lot on a short series"
