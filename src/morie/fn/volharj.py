# morie.fn -- function file (rootcoder007/morie)
"""HAR-RV-J: HAR with a separate jump regressor."""

from . import _array_core as np

from ._richresult import RichResult
from .volhar import _har_design

__all__ = ["vol_har_rv_jump"]


def vol_har_rv_jump(RV, BPV):
    r"""Andersen-Bollerslev-Diebold HAR-RV-J regression.

    .. math:: RV_t = c + \beta_d RV_{t-1} + \beta_w RV^{(w)}_{t-1}
              + \beta_m RV^{(m)}_{t-1} + \beta_J J_{t-1}
              + \varepsilon_t,
              \qquad J_t = \max(RV_t - BPV_t, 0),

    separating yesterday's jump component because jumps are far less
    persistent than diffusive volatility -- empirically
    :math:`\beta_J` is small or negative while the continuous
    cascade carries the forecast.

    Parameters
    ----------
    RV : array-like, shape (n,), n >= 30
        Daily realised variance.
    BPV : array-like, shape (n,)
        Daily bipower variation.

    Returns
    -------
    RichResult
        keys: ``coefficients`` (c, beta_d, beta_w, beta_m, beta_J),
        ``r2``, ``fitted``, ``jump`` (the J series), ``n_obs``,
        ``method``.

    References
    ----------
    Andersen, T. G., Bollerslev, T. & Diebold, F. X. (2007). Roughing
    it up: including jump components in the measurement, modeling,
    and forecasting of return volatility. *The Review of Economics
    and Statistics*, 89(4), 701-720. (HAR-RV-J)

    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-37. (the jump measure)
    """
    RV = np.asarray(RV, dtype=float).ravel()
    BPV = np.asarray(BPV, dtype=float).ravel()
    if RV.size != BPV.size:
        raise ValueError("RV and BPV must have equal length.")
    if RV.size < 30:
        raise ValueError(f"need at least 30 observations, got {RV.size}.")
    if np.any(RV < 0) or np.any(BPV < 0):
        raise ValueError("RV and BPV cannot be negative.")

    J = np.maximum(RV - BPV, 0.0)
    X, y = _har_design(RV)
    Xj = np.column_stack([X, J[21:-1]])
    b, *_ = np.linalg.lstsq(Xj, y, rcond=None)
    fitted = Xj @ b
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return RichResult(
        payload={
            "coefficients": b.astype(float),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "fitted": fitted,
            "jump": J,
            "n_obs": int(y.size),
            "method": "HAR-RV-J (separate lagged-jump regressor)",
        }
    )


def cheatsheet():
    return "volharj: HAR + beta_J max(RV - BPV, 0)_{t-1} (ABD 2007)"


# compact alias per ledger/NAMING.md
volharrvjump = vol_har_rv_jump
