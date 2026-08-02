# morie.fn -- function file (rootcoder007/morie)
"""HAR-Q: HAR-RV with a realised-quarticity attenuation term."""

from . import _array_core as np

from ._richresult import RichResult
from .volhar import _har_design

__all__ = ["vol_har_q"]


def vol_har_q(RV, RQ):
    r"""Bollerslev-Patton-Quaedvlieg HAR-Q regression.

    .. math:: RV_t = c + \big(\beta_d + \beta_q \sqrt{RQ_{t-1}}\big)
              RV_{t-1} + \beta_w RV^{(w)}_{t-1}
              + \beta_m RV^{(m)}_{t-1} + \varepsilon_t.

    The daily coefficient shrinks on days when RV was measured
    noisily -- realised quarticity proxies the measurement-error
    variance -- so a typically *negative* :math:`\beta_q` attenuates
    yesterday's RV exactly when it deserves less trust.

    Parameters
    ----------
    RV : array-like, shape (n,), n >= 30
        Daily realised variance.
    RQ : array-like, shape (n,)
        Daily realised quarticity, same length.

    Returns
    -------
    RichResult
        keys: ``coefficients`` (c, beta_d, beta_w, beta_m, beta_q),
        ``r2``, ``fitted``, ``n_obs``, ``method``.

    References
    ----------
    Bollerslev, T., Patton, A. J. & Quaedvlieg, R. (2016). Exploiting
    the errors: a simple approach for improved volatility
    forecasting. *Journal of Econometrics*, 192(1), 1-18. (HAR-Q)

    Corsi, F. (2009). A simple approximate long-memory model of
    realized volatility. *Journal of Financial Econometrics*, 7(2),
    174-196.
    """
    RV = np.asarray(RV, dtype=float).ravel()
    RQ = np.asarray(RQ, dtype=float).ravel()
    if RV.size != RQ.size:
        raise ValueError("RV and RQ must have equal length.")
    if RV.size < 30:
        raise ValueError(f"need at least 30 observations, got {RV.size}.")
    if np.any(RV < 0) or np.any(RQ < 0):
        raise ValueError("RV and RQ cannot be negative.")

    X, y = _har_design(RV)
    q = np.sqrt(RQ[21:-1])  # aligned with the RV_{t-1} column
    Xq = np.column_stack([X, q * X[:, 1]])
    b, *_ = np.linalg.lstsq(Xq, y, rcond=None)
    fitted = Xq @ b
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return RichResult(
        payload={
            "coefficients": b.astype(float),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "fitted": fitted,
            "n_obs": int(y.size),
            "method": "HAR-Q (quarticity-attenuated daily coefficient)",
        }
    )


def cheatsheet():
    return "volhar1: HAR + beta_q sqrt(RQ_{t-1}) RV_{t-1} (BPQ 2016)"
