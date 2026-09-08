# morie.fn -- function file (rootcoder007/morie)
"""Holt-Winters seasonal exponential smoothing."""

from . import _array_core as np

from ._richresult import RichResult

def _squash(x):
    """Logistic mapped into (eps, 1 - eps).

    A plain sigmoid saturates to exactly 1.0 in float64 once the
    argument passes ~37, which then trips the (0, 1) validity check on
    a perfectly reasonable fit.
    """
    return 1e-6 + (1 - 2e-6) / (1 + np.exp(-np.clip(x, -30, 30)))


__all__ = ["joseph_holt_winters"]


def joseph_holt_winters(y, alpha=None, beta=None, gamma=None, m=12, horizon=1,
                        seasonal="additive"):
    r"""Holt-Winters seasonal method.

    Additive form:

    .. math::

       \ell_t &= \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})\\
       b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1} \\
       s_t &= \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m} \\
       \hat y_{t+h} &= \ell_t + h b_t + s_{t+h-m(k+1)}

    The multiplicative form replaces the seasonal additions with
    ratios, which is the right choice when seasonal swings grow with
    the level -- and is rejected outright for series containing zero or
    negative values rather than silently producing infinities.

    Seasonal indices are initialised from period averages and, in the
    additive case, centred to sum to zero, so the level is not
    confounded with a seasonal offset.

    Parameters
    ----------
    y : array-like
        Series, at least ``2 * m`` observations.
    alpha, beta, gamma : float in (0, 1), optional
        Smoothing parameters; estimated by one-step SSE if omitted.
    m : int, default 12
        Seasonal period.
    horizon : int, default 1
        Forecast horizon.
    seasonal : {"additive", "multiplicative"}
        Seasonal form.

    Returns
    -------
    RichResult
        keys: ``forecast``, ``level``, ``trend``, ``season``,
        ``fitted``, ``residuals``, ``sse``, ``alpha``, ``beta``,
        ``gamma``, ``m``, ``seasonal``, ``n``, ``method``.

    References
    ----------
    Winters, P. R. (1960). Forecasting sales by exponentially weighted
    moving averages. *Management Science*, 6(3), 324-342.

    Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting:
    Principles and Practice* (3rd ed.). OTexts. Sec. 8.3
    (Holt-Winters' seasonal method).
    """
    from ._sci_core import optimize

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    m = int(m)
    if m < 2:
        raise ValueError(f"m must be at least 2, got {m}.")
    if n < 2 * m:
        raise ValueError(f"need at least 2*m = {2 * m} observations, got {n}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite.")
    if seasonal not in ("additive", "multiplicative"):
        raise ValueError("seasonal must be 'additive' or 'multiplicative'.")
    mult = seasonal == "multiplicative"
    if mult and np.any(y <= 0):
        raise ValueError(
            "multiplicative seasonality needs strictly positive data; "
            "use seasonal='additive' for series with zeros or negatives."
        )
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be at least 1, got {h}.")

    k = n // m
    periods = y[: k * m].reshape(k, m)
    overall = float(periods.mean())

    # Initial seasonal indices from a CENTRED MOVING AVERAGE detrend,
    # not from raw period means: with a trend inside the period, the
    # per-position mean absorbs 0.5 * position of slope and the
    # resulting indices come out phase-shifted (Hyndman &
    # Athanasopoulos Sec. 3.4, classical decomposition).
    half = m // 2
    trend_ma = np.full(n, np.nan)
    if m % 2 == 0:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    else:
        w = np.ones(m) / m
    valid = np.convolve(y, w, mode="valid")
    start = (w.size - 1) // 2
    trend_ma[start : start + valid.size] = valid
    detr = (y / trend_ma) if mult else (y - trend_ma)
    s0 = np.empty(m)
    for j in range(m):
        vals = detr[j::m]
        vals = vals[np.isfinite(vals)]
        s0[j] = np.mean(vals) if vals.size else (1.0 if mult else 0.0)
    if mult:
        s0 = s0 / s0.mean()
    else:
        s0 = s0 - s0.mean()  # centre so the level is identified

    def run(a, b, g):
        lev = np.empty(n)
        tr = np.empty(n)
        se = np.empty(n + m)
        fit = np.empty(n)
        se[:m] = s0
        lev[0] = overall
        tr[0] = (periods[-1].mean() - periods[0].mean()) / max((k - 1) * m, 1)
        fit[0] = lev[0] * se[0] if mult else lev[0] + se[0]
        for t in range(1, n):
            p = lev[t - 1] + tr[t - 1]
            fit[t] = p * se[t] if mult else p + se[t]
            if mult:
                lev[t] = a * (y[t] / se[t]) + (1 - a) * p
                se[t + m] = g * (y[t] / max(lev[t], 1e-12)) + (1 - g) * se[t]
            else:
                lev[t] = a * (y[t] - se[t]) + (1 - a) * p
                se[t + m] = g * (y[t] - p) + (1 - g) * se[t]
            tr[t] = b * (lev[t] - lev[t - 1]) + (1 - b) * tr[t - 1]
        return lev, tr, se, fit

    if alpha is None or beta is None or gamma is None:
        def sse(x):
            a, b, g = _squash(x)
            _, _, _, f = run(a, b, g)
            r = y[m:] - f[m:]
            return float(np.sum(r**2)) if np.all(np.isfinite(f)) else 1e18

        res = optimize.minimize(sse, [0.0, -2.0, -1.0], method="Nelder-Mead",
                                options={"maxiter": 800})
        ah, bh, gh = _squash(res.x)
        alpha = ah if alpha is None else float(alpha)
        beta = bh if beta is None else float(beta)
        gamma = gh if gamma is None else float(gamma)
    alpha, beta, gamma = float(alpha), float(beta), float(gamma)
    for nm, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not 0 < v < 1:
            raise ValueError(f"{nm} must lie in (0, 1), got {v}.")

    lev, tr, se, fit = run(alpha, beta, gamma)
    # se[t] is the index in force at time t and se has length n + m,
    # so the horizon-i forecast uses se[n + i] directly (it was last
    # updated at t = n + i - m). Reaching back another period would
    # apply a stale index one season out of phase.
    sf = np.array([se[n + i] if n + i < se.size else se[n + i - m] for i in range(h)])
    base = lev[-1] + np.arange(1, h + 1) * tr[-1]
    return RichResult(
        payload={
            "forecast": base * sf if mult else base + sf,
            "level": lev, "trend": tr, "season": se[:n], "fitted": fit,
            "residuals": y - fit, "sse": float(np.sum((y[m:] - fit[m:]) ** 2)),
            "alpha": alpha, "beta": beta, "gamma": gamma, "m": m,
            "seasonal": seasonal, "n": int(n),
            "method": f"Holt-Winters {seasonal} seasonal method (m = {m})",
        }
    )


def cheatsheet():
    return "johw: level+trend+season; multiplicative refuses non-positive data"
