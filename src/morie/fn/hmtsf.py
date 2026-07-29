# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time series forecasting with RNN."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_time_series_forecast"]


def geron_time_series_forecast(y, horizon=1, window=3, ridge=0.0, recursive=True):
    """
    Time series forecasting with a fixed-width lag window.

    Formula: y_{T+h} = f(y_{T-w+1}, ..., y_T)

    Builds the supervised windowing that any sequence model (RNN, CNN or
    linear) trains on -- rows ``(y_{t-w+1..t}) -> y_{t+1}`` -- and fits the
    linear read-out in closed form (ridge-regularised normal equations),
    then rolls the one-step model forward `horizon` times feeding its own
    predictions back in. That is the same recursive protocol an RNN
    forecaster uses at inference, so the naive-persistence comparison in
    the payload is a real skill score, not a claim.

    Parameters
    ----------
    y : array-like
        Univariate series, length >= window + 2.
    horizon : int, default 1
        Steps ahead to forecast (>= 1).
    window : int, default 3
        Lag window width (>= 1).
    ridge : float, default 0.0
        L2 penalty on the lag weights (not the intercept); >= 0.
    recursive : bool, default True
        True: iterate the one-step model. False: fit one direct model per
        horizon step (no error compounding, needs more data).

    Returns
    -------
    result : RichResult
        Keys: forecast, coef, intercept, train_mse, naive_mse, skill,
        estimate, n, method.

    Examples
    --------
    A linear ramp satisfies y_t = 2*y_{t-1} - y_{t-2} exactly, so a
    window of 2 extrapolates it with no error:

    >>> r = geron_time_series_forecast([1.0, 2, 3, 4, 5, 6, 7, 8], horizon=3, window=2)
    >>> [round(float(v), 6) for v in r["forecast"]]
    [9.0, 10.0, 11.0]
    >>> round(float(r["train_mse"]), 12)
    0.0

    A constant series is forecast as the constant:

    >>> [round(float(v), 6) for v in geron_time_series_forecast([5.0] * 8, horizon=2, window=2)["forecast"]]
    [5.0, 5.0]

    References
    ----------
    Géron Ch 13
    """
    s = np.asarray(y, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("geron_time_series_forecast: y is empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("geron_time_series_forecast: y contains non-finite values")
    w = int(window)
    h = int(horizon)
    if w < 1:
        raise ValueError(f"geron_time_series_forecast: window must be >= 1, got {w}")
    if h < 1:
        raise ValueError(f"geron_time_series_forecast: horizon must be >= 1, got {h}")
    lam = float(ridge)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("geron_time_series_forecast: ridge must be a non-negative finite penalty")
    steps = 1 if recursive else h
    need = w + steps + 1
    if s.size < need:
        raise ValueError(
            f"geron_time_series_forecast: need at least {need} observations for window={w} "
            f"and horizon={h} ({'recursive' if recursive else 'direct'}), got {s.size}"
        )

    def _fit(lead):
        rows = s.size - w - lead + 1
        A = np.empty((rows, w + 1))
        t = np.empty(rows)
        for i in range(rows):
            A[i, :w] = s[i : i + w]
            A[i, w] = 1.0
            t[i] = s[i + w + lead - 1]
        P = np.eye(w + 1) * lam
        P[w, w] = 0.0
        # Least-norm solve: a constant or perfectly collinear window makes the
        # normal matrix singular, and the min-norm fit is still the right answer.
        beta = np.linalg.lstsq(A.T @ A + P, A.T @ t, rcond=None)[0]
        return beta, A, t

    if recursive:
        beta, A, t = _fit(1)
        hist = list(s[-w:])
        fc = []
        for _ in range(h):
            nxt = float(np.dot(beta[:w], hist[-w:]) + beta[w])
            fc.append(nxt)
            hist.append(nxt)
        coef = beta[:w]
        intercept = float(beta[w])
        resid = A @ beta - t
        train_mse = float(resid @ resid / resid.size)
    else:
        fc = []
        coef = []
        train_sse = 0.0
        train_n = 0
        for lead in range(1, h + 1):
            beta, A, t = _fit(lead)
            fc.append(float(np.dot(beta[:w], s[-w:]) + beta[w]))
            coef.append(beta[:w])
            r = A @ beta - t
            train_sse += float(r @ r)
            train_n += r.size
        coef = np.asarray(coef)
        intercept = None
        train_mse = train_sse / train_n

    naive = float(np.mean((s[w:] - s[w - 1 : -1]) ** 2)) if s.size > w else float("nan")
    skill = float(1.0 - train_mse / naive) if naive > 0 else float("nan")

    return RichResult(
        title="Lag-window time series forecast",
        summary_lines=[
            ("Window", w),
            ("Horizon", h),
            ("Train MSE", train_mse),
            ("Naive (persistence) MSE", naive),
        ],
        interpretation=(
            "A forecast is only useful if it beats persistence; skill > 0 means the lag window carries "
            "information beyond 'tomorrow looks like today'."
        ),
        payload={
            "forecast": np.asarray(fc, dtype=float),
            "coef": np.asarray(coef, dtype=float),
            "intercept": intercept,
            "train_mse": train_mse,
            "naive_mse": naive,
            "skill": skill,
            "window": w,
            "horizon": h,
            "estimate": float(fc[-1]),
            "n": int(s.size),
            "method": ("Recursive" if recursive else "Direct") + " lag-window linear forecast (ridge normal equations)",
        },
    )


def cheatsheet():
    return "hmtsf: Time series forecasting with RNN"
