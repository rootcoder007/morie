# morie.fn -- function file (rootcoder007/morie)
"""HAR-RV: heterogeneous autoregressive model of realised volatility."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_har_rv", "_har_design"]


def _har_design(RV):
    """Daily/weekly/monthly HAR regressors aligned to predict RV[t]."""
    RV = np.asarray(RV, dtype=float).ravel()
    n = RV.size
    rows = []
    ys = []
    for t in range(22, n):
        d = RV[t - 1]
        w = RV[t - 5 : t].mean()
        m = RV[t - 22 : t].mean()
        rows.append([1.0, d, w, m])
        ys.append(RV[t])
    return np.array(rows), np.array(ys)


def vol_har_rv(RV, h=1):
    r"""Corsi's HAR-RV regression.

    .. math:: RV_t = c + \beta_d RV_{t-1} + \beta_w RV_{t-1}^{(w)}
              + \beta_m RV_{t-1}^{(m)} + \varepsilon_t,

    with weekly and monthly regressors the 5- and 22-day trailing
    means -- an additive cascade of heterogeneous horizons that
    reproduces the long-memory feel of volatility with three plain
    OLS coefficients. ``h`` steps ahead are forecast by iterating the
    fitted equation.

    Parameters
    ----------
    RV : array-like, shape (n,), n >= 30
        Daily realised variance series.
    h : int, default 1
        Forecast horizon (days ahead).

    Returns
    -------
    RichResult
        keys: ``coefficients`` (c, beta_d, beta_w, beta_m), ``r2``,
        ``fitted``, ``forecast`` (h,), ``n_obs``, ``method``.

    References
    ----------
    Corsi, F. (2009). A simple approximate long-memory model of
    realized volatility. *Journal of Financial Econometrics*, 7(2),
    174-196.
    """
    RV = np.asarray(RV, dtype=float).ravel()
    if RV.size < 30:
        raise ValueError(f"need at least 30 observations, got {RV.size}.")
    if np.any(RV < 0):
        raise ValueError("realised variance cannot be negative.")
    h = int(h)
    if h < 1:
        raise ValueError(f"h must be at least 1, got {h}.")

    X, y = _har_design(RV)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ b
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # iterate for the forecast path
    hist = RV.copy()
    fc = []
    for _ in range(h):
        d = hist[-1]
        w = hist[-5:].mean()
        m = hist[-22:].mean()
        nxt = float(b[0] + b[1] * d + b[2] * w + b[3] * m)
        nxt = max(nxt, 0.0)
        fc.append(nxt)
        hist = np.append(hist, nxt)

    return RichResult(
        payload={
            "coefficients": b.astype(float),
            "r2": float(r2),
            "fitted": fitted,
            "forecast": np.array(fc),
            "n_obs": int(y.size),
            "method": "HAR-RV (daily + weekly + monthly cascade, OLS)",
        }
    )


def cheatsheet():
    return "volhar: RV_t ~ c + b_d RV_{t-1} + b_w mean5 + b_m mean22 (Corsi 2009)"


# compact alias per ledger/NAMING.md
volharrv = vol_har_rv
