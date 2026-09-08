# morie.fn -- function file (rootcoder007/morie)
"""Holt-Winters multiplicative seasonal method."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["holt_winters_mult"]


def holt_winters_mult(y, period=4, alpha=0.3, beta=0.1, gamma=0.1, horizon=None):
    r"""Triple exponential smoothing with multiplicative seasonality.

    Level, trend and seasonal components each get their own smoothing
    parameter:

    .. math::
        \ell_t &= \alpha\,(y_t / s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
        b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1} \\
        s_t &= \gamma\,(y_t / \\ell_t) + (1-\gamma)s_{t-m}

    with forecast :math:`\hat y_{T+h} = (\\ell_T + h b_T)\\,s_{T+h-m}`.

    Multiplicative seasonality means the seasonal swing is a **percentage of
    the level** -- December is 40% above trend whether the trend is at 1000 or
    10000. This is the right form for most commercial series, where seasonal
    amplitude grows with the business.

    It is undefined at or below zero, since the seasonal index divides the
    data, and that is enforced rather than left to produce silent nonsense. A
    log transform plus the additive form is the usual alternative when the
    series has occasional zeros.

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
    On a series whose seasonal amplitude grows with the level, the
    multiplicative form fits better than the additive one -- which is the
    entire reason to choose it.

    >>> import numpy as np
    >>> from morie.fn.hwadd import holt_winters_additive
    >>> t = np.arange(72)
    >>> y = (100 + 3 * t) * (1 + 0.4 * np.sin(2 * np.pi * t / 12))
    >>> mul = holt_winters_mult(y, period=12, alpha=0.4, beta=0.1, gamma=0.3)
    >>> add = holt_winters_additive(y, period=12, alpha=0.4, beta=0.1, gamma=0.3)
    >>> bool(mul["sse"] < add["sse"])
    True

    Seasonal indices centre on 1 rather than on 0, since they multiply.

    >>> bool(0.4 < float(np.mean(mul["seasonal"])) < 1.6)
    True

    Non-positive data is refused outright, since the index divides the series.

    >>> holt_winters_mult(np.r_[np.ones(24), 0.0], period=12)
    Traceback (most recent call last):
        ...
    ValueError: multiplicative seasonality needs strictly positive data
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

    MULT = 'multiplicative' == "multiplicative"
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
        title="Holt-Winters (multiplicative)",
        summary_lines=[("n", int(y.size)), ("period", m),
                       ("level", float(level)), ("trend", float(trend))],
        payload={
            "forecast": fc, "level": float(level), "trend": float(trend),
            "seasonal": np.array(seas[-m:]), "fitted": fitted,
            "residuals": resid, "sse": float(np.sum(resid**2)),
            "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "period": m, "horizon": horizon, "method": "holt_winters_mult",
        },
    )


def cheatsheet():
    return "hwmul: multiplicative season; needs 2+ full periods to initialise, 3 params is a lot on a short series"
