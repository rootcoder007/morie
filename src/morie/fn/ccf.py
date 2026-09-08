# morie.fn -- function file (rootcoder007/morie)
"""Cross-correlation function between two time series."""

from . import _array_core as np
from . import _stats_core as _stats


def ccf(
    x: np.ndarray,
    y: np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05,
) -> dict:
    r"""
    Cross-correlation function between two time series.

    Computes the Pearson correlation between *x* and lagged *y*:

    .. math::

        \\hat{\\rho}_{xy}(h) = \\frac{\\sum_{t} (x_t - \\bar{x})(y_{t+h} - \\bar{y})}
            {\\sqrt{\\sum (x_t - \\bar{x})^2 \\sum (y_t - \\bar{y})^2}}

    for lags :math:`h = -\\text{nlags}, \\ldots, +\\text{nlags}`.

    :param x: 1-D array (reference series).
    :param y: 1-D array (lagged series). Same length as *x*.
    :param nlags: Maximum number of lags in each direction. Default 20.
    :param alpha: Significance level for the approximate CI
        (Bartlett white-noise bound). Default 0.05.
    :return: dict with ``lags`` (array), ``ccf_values`` (array),
        ``ci`` (float, half-width of white-noise CI).
    :raises ValueError: If *x* and *y* differ in length or are too short.

    References
    ----------
    Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (2015). Time Series
    Analysis: Forecasting and Control (5th ed.). Wiley.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}.")
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}.")
    nlags = min(nlags, n - 1)

    xm = x - np.mean(x)
    ym = y - np.mean(y)
    sx = np.sqrt(np.sum(xm**2))
    sy = np.sqrt(np.sum(ym**2))
    denom = sx * sy
    if denom == 0:
        raise ValueError("One or both series have zero variance.")

    # np.arange yields floats here, and a float cannot slice -- every
    # negative lag raised TypeError, so this function could not run at all.
    lag_ints = list(range(-nlags, nlags + 1))
    lags_arr = np.array([float(h) for h in lag_ints])
    ccf_vals = np.zeros(len(lag_ints))

    for i, h in enumerate(lag_ints):
        if h >= 0:
            ccf_vals[i] = np.sum(xm[: n - h] * ym[h:]) / denom
        else:
            ccf_vals[i] = np.sum(xm[-h:] * ym[: n + h]) / denom

    z = _stats.norm.ppf(1 - alpha / 2)
    ci = z / np.sqrt(n)

    return {
        "lags": lags_arr,
        "ccf_values": ccf_vals,
        "ci": float(ci),
    }


def cheatsheet() -> str:
    return "ccf({}) -> Cross-correlation function between two time series."


def cross_correlation(x, y, max_lag: int | None = None):
    """Normalized cross-correlation with the lag axis running from +max_lag
    down to -max_lag.

    This was fn/xcorr.py, a second implementation of :func:`ccf` above
    producing the same numbers with the lag axis REVERSED -- the opposite
    sign convention for the delay, which is exactly how a delay estimate
    comes out with the wrong sign.  It now delegates, so there is one
    implementation and the reversal is explicit.

    The bare name ``xcorr`` no longer points here: in signal processing
    xcorr means the RAW cross-correlation, which is
    :func:`morie.fn.bsacorr.xcorr` (Rangayyan's R_xy(m)).
    """
    from ._containers import DescriptiveResult

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)
    nlags = (n - 1) if max_lag is None else int(max_lag)
    nlags = max(1, min(nlags, n - 1))
    corr = np.array(list(ccf(xa, ya, nlags=nlags)["ccf_values"])[::-1])
    return DescriptiveResult(
        name="cross_correlation",
        value=float(np.max(np.abs(corr))),
        extra={"correlation": corr, "max_lag": max_lag,
               "lag_axis_runs_positive_to_negative": True},
    )
