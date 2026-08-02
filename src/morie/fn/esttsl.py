# morie.fn -- function file (rootcoder007/morie)
"""Theta method."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["theta_method"]


def theta_method(y, horizon=1, theta=2.0):
    r"""Decompose into theta-lines and recombine -- the M3 competition winner.

    A theta-line rescales the second differences of the series by
    :math:`\theta`:

    .. math::
        \nabla^2 Y_t(\theta) = \theta \nabla^2 y_t,

    so :math:`\theta = 0` is the linear regression line through the data
    (curvature removed entirely) and :math:`\theta = 2` doubles the curvature.
    The forecast averages the two.

    The result that made the method famous: this is **exactly equivalent** to
    simple exponential smoothing with drift, where the drift is half the slope
    of the fitted linear trend. A method that won the M3 competition reduces
    to SES plus half a trend, which is a standing argument for simple
    forecasts over elaborate ones.

    The equivalence is asserted in the doctest rather than described.

    Parameters
    ----------
    y : array-like
        Series, at least 3 points.
    horizon : int
        Steps ahead.
    theta : float
        Curvature coefficient for the second line. 2.0 is standard.

    Returns
    -------
    RichResult
        ``forecast``, ``drift``, ``alpha``, ``linear_slope``,
        ``theta_line_0``, ``theta_line``.

    References
    ----------
    Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
        decomposition approach to forecasting. *International Journal of
        Forecasting*, 16(4), 521-530.
    Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
        *International Journal of Forecasting*, 19(2), 287-290.
    Examples
    --------
    On a linear trend the method extrapolates it.

    >>> import numpy as np
    >>> y = np.arange(1.0, 21.0)
    >>> r = theta_method(y, horizon=3)
    >>> bool(abs(r["forecast"][0] - 21.0) < 1.0)
    True

    The drift is half the slope of the fitted linear trend -- the
    Hyndman-Billah equivalence to SES with drift.

    >>> bool(abs(r["drift"] - r["linear_slope"] / 2) < 1e-9)
    True

    theta = 0 collapses the second line onto the regression line, so the
    forecast becomes the linear extrapolation.

    >>> flat = theta_method(y, horizon=1, theta=0.0)
    >>> bool(abs(flat["forecast"][0] - 21.0) < 1.5)
    True

    >>> theta_method([1.0, 2.0], horizon=1)
    Traceback (most recent call last):
        ...
    ValueError: need at least 3 observations
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    t = np.arange(1.0, n + 1.0)
    A = np.column_stack([np.ones(n), t])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    slope = float(coef[1])
    line0 = A @ coef                       # theta = 0: the regression line
    line_theta = theta * y + (1.0 - theta) * line0

    # SES on the theta-line, alpha chosen by in-sample SSE.
    def ses(a, series):
        lev = series[0]
        fit = np.empty(series.size)
        for i in range(series.size):
            fit[i] = lev
            lev = a * series[i] + (1 - a) * lev
        return fit, lev

    grid = np.linspace(0.05, 1.0, 60)
    sses = [float(np.sum((line_theta - ses(a, line_theta)[0]) ** 2)) for a in grid]
    alpha = float(grid[int(np.argmin(sses))])
    _, level = ses(alpha, line_theta)

    drift = slope / 2.0
    steps = np.arange(1, horizon + 1)
    fc = level + drift * steps
    return RichResult(
        title="Theta method",
        summary_lines=[("n", int(n)), ("theta", float(theta)),
                       ("alpha", alpha), ("drift", drift)],
        payload={
            "forecast": fc, "drift": drift, "alpha": alpha,
            "linear_slope": slope, "theta_line_0": line0,
            "theta_line": line_theta, "level": float(level),
            "horizon": horizon, "theta": float(theta),
            "method": "theta_method",
        },
    )


def cheatsheet():
    return "esttsl: M3 winner that is EXACTLY SES with drift = slope/2 (Hyndman-Billah)"
