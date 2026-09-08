# morie.fn -- function file (rootcoder007/morie)
"""Augmented Synthetic Control (Ben-Michael-Feller-Rothstein)."""

from . import _array_core as np

from ._richresult import RichResult
from .caussc import causal_synthetic_control

__all__ = ["augmented_synthetic_control"]


def augmented_synthetic_control(y_treated, y_controls, treat_time, ridge=1e-3):
    r"""Ridge-augmented SCM: bias-correct the post gap with an outcome model.

    Ben-Michael, Feller and Rothstein's estimator corrects the SCM gap
    for imperfect pre-period fit:

    .. math:: \hat Y_{1t}(0) = \sum_j w_j Y_{jt}
              + \Big( \hat m_t(X_1) - \sum_j w_j \hat m_t(X_j) \Big),

    where :math:`\hat m_t` is an outcome model (here ridge regression
    of the period-t donor outcome on donor pre-period paths). When the
    SCM weights already balance the pre-period exactly, the correction
    vanishes; when they cannot, it extrapolates -- trading pure
    interpolation for lower bias.

    Parameters
    ----------
    y_treated : array-like, shape (T,)
        Treated unit's outcome series.
    y_controls : array-like, shape (T, J)
        Donor outcomes.
    treat_time : int
        First post-treatment index.
    ridge : float, default 1e-3
        Ridge penalty for the outcome model.

    Returns
    -------
    RichResult
        keys: ``att`` (augmented), ``att_scm`` (unaugmented),
        ``correction``, ``weights``, ``rmse_pre``, ``treat_time``,
        ``method``.

    References
    ----------
    Ben-Michael, E., Feller, A. & Rothstein, J. (2021). The augmented
    synthetic control method. *Journal of the American Statistical
    Association*, 116(536), 1789-1803.
    """
    y1 = np.asarray(y_treated, dtype=float).ravel()
    Y0 = np.asarray(y_controls, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y1.size:
        raise ValueError("y_controls must be (T, J) matching y_treated.")
    t0 = int(treat_time)
    if not 2 <= t0 < y1.size:
        raise ValueError(f"treat_time must lie in [2, T), got {t0}.")
    ridge = float(ridge)
    if ridge <= 0:
        raise ValueError(f"ridge must be positive, got {ridge}.")

    fit = causal_synthetic_control(y1[:t0], Y0[:t0])
    w = fit["weights"]

    # outcome model per post period: ridge of Y_{jt} on donor pre paths
    Xd = Y0[:t0].T  # (J, t0): donor pre-period paths as features
    Xd_c = Xd - Xd.mean(axis=0)
    G = Xd_c.T @ Xd_c + ridge * np.eye(t0)
    x1_c = y1[:t0] - Xd.mean(axis=0)

    att_scm = float((y1[t0:] - Y0[t0:] @ w).mean())
    corr = 0.0
    for t in range(t0, y1.size):
        yt = Y0[t]
        beta = np.linalg.solve(G, Xd_c.T @ (yt - yt.mean()))
        m1 = yt.mean() + x1_c @ beta
        mw = yt.mean() + (Xd_c @ beta) @ w
        corr += m1 - mw
    corr /= y1.size - t0

    return RichResult(
        payload={
            "att": att_scm - corr,
            "att_scm": att_scm,
            "correction": float(corr),
            "weights": w,
            "rmse_pre": fit["rmse_pre"],
            "treat_time": t0,
            "method": "Augmented Synthetic Control (ridge outcome model)",
        }
    )


def cheatsheet():
    return "ascmcl: SCM gap minus ridge outcome-model correction (Ben-Michael et al. 2021)"
