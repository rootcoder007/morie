# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ARIMA(p, d, q) one-step-ahead forecast (post-differencing)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_arima_forecast"]

_METHOD = "ARIMA(p, d, q) one-step-ahead forecast"


def geron_arima_forecast(y, phi, theta, d=0):
    r"""One-step-ahead ARIMA forecast with known coefficients.

    The series is differenced ``d`` times to give
    :math:`w_t = (1-B)^d y_t`, the innovations are recovered by the usual
    recursion with zero pre-sample values,

    .. math::
        \varepsilon_t = w_t - \sum_{i=1}^{p}\phi_i w_{t-i}
                            - \sum_{j=1}^{q}\theta_j \varepsilon_{t-j},

    the ARMA forecast is formed on the differenced scale,

    .. math::
        \hat w_{T+1} = \sum_{i=1}^{p}\phi_i w_{T+1-i}
                     + \sum_{j=1}^{q}\theta_j \varepsilon_{T+1-j},

    and the differencing is then undone by adding back the last value of
    each lower-order difference.

    Parameters
    ----------
    y : array-like
        Observed series in time order.
    phi : array-like
        AR coefficients :math:`\phi_1 \dots \phi_p` (may be empty).
    theta : array-like
        MA coefficients :math:`\theta_1 \dots \theta_q` (may be empty),
        in the *plus* sign convention shown above.
    d : int, optional
        Order of differencing, default 0.

    Returns
    -------
    RichResult
        Payload keys ``forecast`` (on the original scale),
        ``forecast_differenced``, ``residuals``, ``differenced``,
        ``sigma2`` (residual variance), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 13, ARMA / ARIMA section.

    Examples
    --------
    A random walk with drift-like AR(1) on the first differences:

    >>> r = geron_arima_forecast([1.0, 2.0, 3.0, 4.0], phi=[0.5], theta=[], d=1)
    >>> r["differenced"]
    [1.0, 1.0, 1.0]
    >>> round(r["forecast_differenced"], 6)
    0.5
    >>> round(r["forecast"], 6)
    4.5

    Pure MA(1) with no differencing:

    >>> r2 = geron_arima_forecast([1.0, 2.0], phi=[], theta=[0.5], d=0)
    >>> [round(e, 6) for e in r2["residuals"]]
    [1.0, 1.5]
    >>> round(r2["forecast"], 6)
    0.75
    """
    y = np.asarray(y, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(theta)):
        raise ValueError("phi and theta must be finite.")
    d = int(d)
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}.")
    p = phi.size
    q = theta.size
    need = d + max(p, 1)
    if y.size < need:
        raise ValueError(
            f"y has {y.size} observations but ARIMA(p={p}, d={d}, q={q}) needs at "
            f"least {need} to form one forecast."
        )

    # Difference d times, remembering the last level at each stage so we
    # can integrate the forecast back up.
    last_levels = []
    w = y
    for _ in range(d):
        last_levels.append(float(w[-1]))
        w = np.diff(w)

    T = w.size
    eps = np.zeros(T, dtype=float)
    for t in range(T):
        ar = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                ar += phi[i - 1] * w[t - i]
        ma = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                ma += theta[j - 1] * eps[t - j]
        eps[t] = w[t] - ar - ma

    fc = 0.0
    for i in range(1, p + 1):
        if T - i >= 0:
            fc += phi[i - 1] * w[T - i]
    for j in range(1, q + 1):
        if T - j >= 0:
            fc += theta[j - 1] * eps[T - j]
    fc_diff = float(fc)

    level = fc_diff
    for lv in reversed(last_levels):
        level += lv

    if not np.isfinite(level):
        raise ValueError("forecast is not finite; check phi/theta for explosive roots.")

    return RichResult(
        title="ARIMA one-step forecast",
        summary_lines=[("Forecast", level), ("Order", f"({p}, {d}, {q})")],
        payload={
            "forecast": level,
            "forecast_differenced": fc_diff,
            "residuals": eps.tolist(),
            "differenced": w.tolist(),
            "sigma2": float(np.mean(eps**2)),
            "order": (p, d, q),
            "estimate": level,
            "n": int(y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grarma: ARIMA(p,d,q) one-step forecast from known phi/theta"
