# morie.fn -- function file (rootcoder007/morie)
"""Time-series outlier detection by rolling robust score."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["joseph_ts_outlier_detection"]


def joseph_ts_outlier_detection(y, W=10, threshold=3.5):
    r"""Flag points far from a rolling median in robust scale units.

    .. math::
        s_t = \\frac{\\left|y_t - \\operatorname{med}(y_{t-W:t+W})\\right|}
                   {1.4826 \\cdot \\operatorname{MAD}(y_{t-W:t+W})}.

    The rolling median is what makes this work on a series with trend or
    seasonality: a global mean would flag every point in a seasonal peak,
    while a local one adapts. The MAD scaling makes the threshold
    interpretable in approximate standard deviations for Gaussian noise --
    the 1.4826 is the consistency constant.

    The window is the trade, and it binds harder than it looks. It must be
    **short relative to the shortest real feature**, not merely short: on a
    series with period 40, a window of 21 points spans half a cycle, the
    seasonal swing inflates the rolling MAD, and a spike twelve noise-sd tall
    scores only 1.9 -- masked by the very structure the rolling reference was
    meant to remove. The doctest measures that decay directly.

    Consecutive outliers defeat this, as they defeat any local-reference
    method: a run longer than half the window becomes the local median and
    the run is declared normal.

    Parameters
    ----------
    y : array-like
        Series.
    W : int
        Half-window for the rolling statistics.
    threshold : float
        Robust-score cut-off.

    Returns
    -------
    RichResult
        ``outlier``, ``score``, ``rolling_median``, ``rolling_mad``,
        ``n_outliers``.

    References
    ----------
    Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median
        absolute deviation. *JASA*, 88(424), 1273-1283.

    Examples
    --------
    A spike on a trending, seasonal series is caught where a global rule
    would flag the seasonal peaks instead.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(400)
    >>> y = 0.05 * t + 5 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.3, 400)
    >>> y[200] += 12
    >>> r = joseph_ts_outlier_detection(y, W=3)
    >>> bool(r["outlier"][200])
    True

    The seasonal peaks are not flagged, because the reference is local.

    >>> bool(r["n_outliers"] < 15)
    True

    The window must be short relative to the seasonal period: as it grows to
    span a meaningful fraction of a cycle, the swing inflates the rolling MAD
    and the same spike is progressively masked.

    >>> [float(round(joseph_ts_outlier_detection(y, W=w)["score"][200], 1))
    ...  for w in (3, 5, 10)]
    [3.8, 2.8, 1.9]

    The documented failure: a long run of outliers becomes the local median
    and hides itself.

    >>> z = y.copy(); z[100:118] += 12
    >>> bool(joseph_ts_outlier_detection(z, W=8)["outlier"][109] == False)
    True
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = y.size
    W = int(W)
    if W < 1:
        raise ValueError("W must be at least 1")
    if n < 2 * W + 1:
        raise ValueError(f"series of length {n} is too short for W={W}")
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        win = y[lo:hi]
        med[i] = np.median(win)
        mad[i] = np.median(np.abs(win - med[i]))
    scale = 1.4826 * mad
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(y - med) / np.where(scale > 0, scale, np.nan)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0)
    out = score > threshold
    return RichResult(
        title="Time-series outlier detection",
        summary_lines=[("n", int(n)), ("window", 2 * W + 1),
                       ("outliers", int(out.sum()))],
        warnings=["a run of consecutive outliers longer than half the window "
                  "becomes the local median and is declared normal"],
        payload={
            "outlier": out, "score": score, "rolling_median": med,
            "rolling_mad": mad, "n_outliers": int(out.sum()),
            "W": W, "threshold": float(threshold),
            "method": "joseph_ts_outlier_detection",
        },
    )


def cheatsheet():
    return "jooutl: rolling median so trend/season do not fire; a RUN longer than half the window hides itself"
