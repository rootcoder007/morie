# morie.fn -- function file (rootcoder007/morie)
"""Simple exponential smoothing."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["joseph_simple_exponential_smoothing"]


def joseph_simple_exponential_smoothing(y, alpha=None, horizon=1):
    r"""Exponentially weighted level, forecast flat.

    .. math::
        \ell_t = \alpha y_t + (1-\alpha)\ell_{t-1},
        \qquad \hat y_{T+h} = \ell_T .

    The forecast is **flat** for every horizon, because SES models a level and
    nothing else. Applying it to trending data produces a forecast that is
    systematically wrong in a direction anyone can see on a plot -- Holt's
    method exists precisely to add the trend term.

    :math:`\alpha` sets the memory: the weight on an observation :math:`k`
    steps back is :math:`\alpha(1-\alpha)^k`, so the effective window is
    about :math:`2/\alpha - 1` observations. :math:`\alpha \to 1` is the naive
    forecast, :math:`\alpha \to 0` is the overall mean. When omitted,
    :math:`\alpha` is chosen by minimising in-sample SSE.

    Parameters
    ----------
    y : array-like
        Series.
    alpha : float, optional
        Smoothing parameter in (0, 1]. Estimated when omitted.
    horizon : int
        Steps ahead.

    Returns
    -------
    RichResult
        ``forecast``, ``level``, ``fitted``, ``residuals``, ``alpha``,
        ``sse``.

    References
    ----------
    Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles
        and Practice* (3rd ed.). OTexts.

    Examples
    --------
    The forecast is flat -- one level repeated for every horizon.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = joseph_simple_exponential_smoothing(rng.normal(10, 1, 100), alpha=0.3,
    ...                                         horizon=5)
    >>> bool(np.allclose(r["forecast"], r["forecast"][0]))
    True

    alpha = 1 recovers the naive forecast exactly.

    >>> y = [1.0, 5.0, 2.0, 8.0]
    >>> float(joseph_simple_exponential_smoothing(y, alpha=1.0)["forecast"][0])
    8.0

    Estimated alpha beats an arbitrary one on in-sample fit, which is what
    estimating it is for.

    >>> fitted = joseph_simple_exponential_smoothing(rng.normal(10, 1, 200))
    >>> bad = joseph_simple_exponential_smoothing(rng.normal(10, 1, 200), alpha=0.99)
    >>> bool(fitted["sse"] <= bad["sse"] * 1.001)
    True

    >>> joseph_simple_exponential_smoothing([1.0, 2.0], alpha=1.5)
    Traceback (most recent call last):
        ...
    ValueError: alpha must be in (0, 1]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if y.size < 2:
        raise ValueError("need at least 2 observations")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    def run(a):
        lev = y[0]
        fit = np.empty(y.size)
        for i in range(y.size):
            fit[i] = lev
            lev = a * y[i] + (1 - a) * lev
        return fit, lev

    if alpha is None:
        grid = np.linspace(0.01, 1.0, 100)
        sses = [float(np.sum((y - run(a)[0]) ** 2)) for a in grid]
        alpha = float(grid[int(np.argmin(sses))])
    else:
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
    fitted, level = run(alpha)
    resid = y - fitted
    return RichResult(
        title="Simple exponential smoothing",
        summary_lines=[("n", int(y.size)), ("alpha", alpha),
                       ("level", float(level))],
        payload={
            "forecast": np.full(horizon, level), "level": float(level),
            "fitted": fitted, "residuals": resid, "alpha": alpha,
            "sse": float(np.sum(resid**2)),
            "effective_window": float(2.0 / alpha - 1.0),
            "horizon": horizon,
            "method": "joseph_simple_exponential_smoothing",
        },
    )


def cheatsheet():
    return "joses: FLAT forecast -- models a level only; use Holt if the series trends"
