# morie.fn -- function file (rootcoder007/morie)
"""Theta method."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["theta_method"]


def _ses_given_alpha(y, alpha, level0=None):
    """Simple exponential smoothing, ets(A,N,N) form: forecast l_{t-1},
    l_t = l_{t-1} + alpha (y_t - l_{t-1}). Without ``level0`` the initial
    level is the one minimising the SSE, in closed form: each one-step
    error is a_t - b_t l_0 with b_t = (1 - alpha)^(t-1), so the SSE is a
    quadratic in l_0."""
    n = len(y)
    a = [0.0] * n  # one-step errors with l_0 = 0
    lev = 0.0
    for t in range(n):
        a[t] = y[t] - lev
        lev += alpha * a[t]
    b = [(1.0 - alpha) ** t for t in range(n)]
    if level0 is None:
        level0 = sum(bt * at for at, bt in zip(a, b)) / sum(bt * bt for bt in b)
    err = [at - bt * level0 for at, bt in zip(a, b)]
    level_n = lev + (1.0 - alpha) ** n * level0
    return sum(e * e for e in err), float(level0), float(level_n)


def _ses_fit(y):
    """alpha in [1e-4, 0.9999] (ets's default bounds) minimising the SSE
    profiled over l_0: a 400-point scan, then golden-section refinement
    of the best bracket to machine precision."""
    lo, hi = 1e-4, 0.9999
    grid = [lo + (hi - lo) * k / 399 for k in range(400)]
    sse = [_ses_given_alpha(y, g)[0] for g in grid]
    k = min(range(400), key=sse.__getitem__)
    a_lo, a_hi = grid[max(k - 1, 0)], grid[min(k + 1, 399)]
    gr = (5**0.5 - 1) / 2
    c, d = a_hi - gr * (a_hi - a_lo), a_lo + gr * (a_hi - a_lo)
    fc, fd = _ses_given_alpha(y, c)[0], _ses_given_alpha(y, d)[0]
    for _ in range(200):
        if a_hi - a_lo < 1e-15:
            break
        if fc < fd:
            a_hi, d, fd = d, c, fc
            c = a_hi - gr * (a_hi - a_lo)
            fc = _ses_given_alpha(y, c)[0]
        else:
            a_lo, c, fc = c, d, fd
            d = a_lo + gr * (a_hi - a_lo)
            fd = _ses_given_alpha(y, d)[0]
    cand = [(sse[k], grid[k]), (fc, c), (fd, d)]
    return min(cand)[1]


def theta_method(y, horizon=1, theta=2.0, alpha=None, level0=None):
    r"""Theta method forecasts (Assimakopoulos & Nikolopoulos 2000) in the
    Hyndman-Billah form.

    The theta-line :math:`\theta y_t + (1-\theta) L_t` rescales the
    curvature of the series around its regression line :math:`L_t`.
    Combining the extrapolated regression line (weight
    :math:`1 - 1/\theta`) with simple exponential smoothing reduces
    exactly to SES of the data plus a damped drift (Hyndman & Billah
    2003):

    .. math::

        \hat y_{n+h} = \ell_n + \frac{\theta - 1}{\theta}\, b_0
        \Big[h - 1 + \frac{1 - (1-\alpha)^n}{\alpha}\Big],

    with :math:`b_0` the least-squares slope on :math:`t = 0, \dots, n-1`
    and :math:`\ell_n` the final SES level. :math:`\theta = 2` is the
    classical method (R ``forecast::thetaf``); :math:`\theta = 1` is SES.

    Parameters
    ----------
    y : array-like
        The series, at least 3 observations.
    horizon : int
        Number of steps ahead.
    theta : float
        Theta, at least 1.
    alpha : float, optional
        SES smoothing weight in (0, 1]. By default it is estimated with
        the initial level by minimum SSE (the Gaussian MLE of ets(A,N,N)).
    level0 : float, optional
        Initial SES level; by default the SSE-optimal one for ``alpha``.

    Returns
    -------
    RichResult
        ``forecast``, ``ses_forecast``, ``drift`` (the trend coefficient
        :math:`(1 - 1/\theta) b_0`), ``alpha``, ``level0``, ``level``,
        ``linear_slope``, ``sse``, ``theta_line_0``, ``theta_line``.

    References
    ----------
    Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
    decomposition approach to forecasting. International Journal of
    Forecasting, 16(4), 521-530.
    Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
    International Journal of Forecasting, 19(2), 287-290.

    Examples
    --------
    With alpha fixed at 1 SES is the random walk, so the forecast is the
    last value plus half the slope per step.

    >>> r = theta_method([1.0, 2.0, 4.0, 5.0], horizon=2, alpha=1.0)
    >>> [round(v, 10) for v in r["forecast"]]
    [5.7, 6.4]

    >>> theta_method([1.0, 2.0], horizon=1)
    Traceback (most recent call last):
        ...
    ValueError: need at least 3 observations
    """
    y = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    n = len(y)
    if n < 3:
        raise ValueError("need at least 3 observations")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    theta = float(theta)
    if theta < 1.0:
        raise ValueError("theta must be at least 1")
    if alpha is not None and not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    tbar = (n - 1) / 2.0
    ybar = sum(y) / n
    b0 = sum((t - tbar) * (v - ybar) for t, v in enumerate(y)) / sum((t - tbar) ** 2 for t in range(n))
    a0 = ybar - b0 * tbar
    line0 = [a0 + b0 * t for t in range(n)]
    line_theta = [theta * v + (1.0 - theta) * lv for v, lv in zip(y, line0)]

    alpha = _ses_fit(y) if alpha is None else float(alpha)
    sse, l0, level = _ses_given_alpha(y, alpha, level0)
    drift = (1.0 - 1.0 / theta) * b0
    damp = (1.0 - (1.0 - alpha) ** n) / alpha
    fc = [level + drift * (h - 1 + damp) for h in range(1, horizon + 1)]
    return RichResult(
        title="Theta method",
        summary_lines=[("n", n), ("theta", theta), ("alpha", alpha), ("drift", drift)],
        payload={
            "forecast": fc,
            "ses_forecast": [level] * horizon,
            "drift": drift,
            "alpha": alpha,
            "level0": l0,
            "level": level,
            "sse": sse,
            "linear_slope": b0,
            "theta_line_0": line0,
            "theta_line": line_theta,
            "horizon": horizon,
            "theta": theta,
            "method": "theta_method",
        },
    )


def cheatsheet():
    return "esttsl: theta method = SES + (1 - 1/theta) b0 [h - 1 + (1 - (1-alpha)^n)/alpha] (Hyndman-Billah)"


# compact alias per ledger/NAMING.md
thetamethod = theta_method
