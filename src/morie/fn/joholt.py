# morie.fn -- function file (rootcoder007/morie)
"""Holt's linear trend exponential smoothing."""

from . import _array_core as np

from ._richresult import RichResult

def _squash(x):
    """Logistic mapped into (eps, 1 - eps).

    A plain sigmoid saturates to exactly 1.0 in float64 once the
    argument passes ~37, which then trips the (0, 1) validity check on
    a perfectly reasonable fit.
    """
    return 1e-6 + (1 - 2e-6) / (1 + np.exp(-np.clip(x, -30, 30)))


__all__ = ["joseph_holt_linear"]


def joseph_holt_linear(y, alpha=None, beta=None, horizon=1, damped=False, phi=0.98):
    r"""Holt's linear (double exponential) smoothing.

    .. math::

       \ell_t &= \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
       b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1} \\
       \hat y_{t+h} &= \ell_t + h b_t

    With ``damped=True`` the trend is multiplied by
    :math:`\sum_{i=1}^{h}\phi^i` instead of h, so long-horizon
    forecasts flatten rather than extrapolating a straight line
    forever -- Hyndman's empirical finding that damping usually
    forecasts better.

    ``alpha`` and ``beta`` are estimated by minimising one-step SSE
    when not supplied.

    Parameters
    ----------
    y : array-like
        Series, at least 5 observations.
    alpha, beta : float in (0, 1), optional
        Level and trend smoothing parameters; estimated if omitted.
    horizon : int, default 1
        Forecast horizon.
    damped : bool, default False
        Damp the trend.
    phi : float in (0, 1], default 0.98
        Damping parameter.

    Returns
    -------
    RichResult
        keys: ``forecast`` (horizon,), ``level``, ``trend``,
        ``fitted``, ``residuals``, ``sse``, ``alpha``, ``beta``,
        ``damped``, ``phi``, ``n``, ``method``.

    References
    ----------
    Holt, C. C. (2004). Forecasting seasonals and trends by
    exponentially weighted moving averages. *International Journal of
    Forecasting*, 20(1), 5-10 (reprint of the 1957 ONR memorandum).

    Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting:
    Principles and Practice* (3rd ed.). OTexts. Sec. 8.2 (trend
    methods).
    """
    from scipy import optimize

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite.")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be at least 1, got {h}.")
    if not 0 < phi <= 1:
        raise ValueError(f"phi must lie in (0, 1], got {phi}.")

    def run(a, b):
        lev = np.empty(n)
        tr = np.empty(n)
        fit = np.empty(n)
        lev[0] = y[0]
        tr[0] = y[1] - y[0]
        fit[0] = y[0]
        for t in range(1, n):
            p = lev[t - 1] + (phi if damped else 1.0) * tr[t - 1]
            fit[t] = p
            lev[t] = a * y[t] + (1 - a) * p
            tr[t] = b * (lev[t] - lev[t - 1]) + (1 - b) * (
                phi if damped else 1.0
            ) * tr[t - 1]
        return lev, tr, fit

    if alpha is None or beta is None:
        def sse(x):
            a, b = _squash(x)
            _, _, f = run(a, b)
            return float(np.sum((y[1:] - f[1:]) ** 2))

        res = optimize.minimize(sse, [0.0, -1.0], method="Nelder-Mead",
                                options={"maxiter": 500})
        a_hat, b_hat = _squash(res.x)
        alpha = a_hat if alpha is None else float(alpha)
        beta = b_hat if beta is None else float(beta)
    alpha, beta = float(alpha), float(beta)
    for nm, v in (("alpha", alpha), ("beta", beta)):
        if not 0 < v < 1:
            raise ValueError(f"{nm} must lie in (0, 1), got {v}.")

    lev, tr, fit = run(alpha, beta)
    steps = (
        np.cumsum(phi ** np.arange(1, h + 1)) if damped else np.arange(1, h + 1, dtype=float)
    )
    return RichResult(
        payload={
            "forecast": lev[-1] + steps * tr[-1], "level": lev, "trend": tr,
            "fitted": fit, "residuals": y - fit,
            "sse": float(np.sum((y[1:] - fit[1:]) ** 2)),
            "alpha": alpha, "beta": beta, "damped": bool(damped),
            "phi": float(phi) if damped else None, "n": int(n),
            "method": "Holt linear trend exponential smoothing"
            + (" (damped)" if damped else ""),
        }
    )


def cheatsheet():
    return "joholt: level+trend recursions; damped trend flattens long horizons"
