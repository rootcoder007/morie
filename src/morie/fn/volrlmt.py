# morie.fn -- function file (rootcoder007/morie)
"""AR(1) on log realised volatility."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_realised_log_vol_ar"]


def vol_realised_log_vol_ar(RV, h=1):
    r"""Log-RV AR(1), the short-memory benchmark HAR is measured against.

    .. math:: \log RV_t = c + \phi \log RV_{t-1} + u_t,

    fit by OLS on logs (which also tames RV's right skew), with the
    h-step forecast mapped back with the lognormal half-variance
    correction :math:`\exp(\hat m + \hat s^2/2)`.

    Parameters
    ----------
    RV : array-like, shape (n,), n >= 10, strictly positive
        Daily realised variance series.
    h : int, default 1
        Forecast horizon.

    Returns
    -------
    RichResult
        keys: ``c``, ``phi``, ``sigma_u``, ``r2``, ``forecast`` (h,
        on the RV scale), ``halflife`` (of log-RV shocks), ``n_obs``,
        ``method``.

    References
    ----------
    Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P.
    (2003). Modeling and forecasting realized volatility.
    *Econometrica*, 71(2), 579-625. (log-RV autoregressions)

    Corsi, F. (2009). A simple approximate long-memory model of
    realized volatility. *Journal of Financial Econometrics*, 7(2),
    174-196. (the HAR that this benchmark motivates)
    """
    RV = np.asarray(RV, dtype=float).ravel()
    n = RV.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    if np.any(RV <= 0):
        raise ValueError("RV must be strictly positive for the log transform.")
    h = int(h)
    if h < 1:
        raise ValueError(f"h must be at least 1, got {h}.")

    y = np.log(RV)
    X = np.column_stack([np.ones(n - 1), y[:-1]])
    b, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    c, phi = float(b[0]), float(b[1])
    resid = y[1:] - X @ b
    s_u = float(resid.std(ddof=2))
    ss_tot = float(((y[1:] - y[1:].mean()) ** 2).sum())
    r2 = 1 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    # iterate the log forecast, cumulate the forecast-error variance
    m = y[-1]
    var = 0.0
    fc = []
    for _ in range(h):
        m = c + phi * m
        var = phi**2 * var + s_u**2
        fc.append(np.exp(m + var / 2.0))
    hl = float(np.log(0.5) / np.log(abs(phi))) if 0 < abs(phi) < 1 else float("inf")

    return RichResult(
        payload={
            "c": c,
            "phi": phi,
            "sigma_u": s_u,
            "r2": float(r2),
            "forecast": np.array(fc),
            "halflife": hl,
            "n_obs": int(n - 1),
            "method": "AR(1) on log RV with lognormal back-transform",
        }
    )


def cheatsheet():
    return "volrlmt: log RV_t = c + phi log RV_{t-1}; forecast exp(m + s^2/2)"
