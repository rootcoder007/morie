# morie.fn -- function file (rootcoder007/morie)
"""Test for treatment effect heterogeneity (best linear predictor)."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["causal_forest_hte_test"]


def causal_forest_hte_test(y, D, cate_predictions, propensity=None):
    r"""Chernozhukov et al.'s best-linear-predictor calibration test.

    Regress the residualised outcome on the centred and the
    demeaned-CATE interactions with the residualised treatment,

    .. math:: Y - \hat m(X) = \alpha\,(D - \hat e)
              + \beta\,(D - \hat e)(\hat\tau(X) - \bar{\hat\tau})
              + \varepsilon,

    where :math:`\hat\tau` are the *out-of-bag* forest predictions.
    Then :math:`\alpha \approx 1` says the forest's average effect is
    calibrated, and :math:`\beta` is the heterogeneity coefficient:
    :math:`\beta = 0` means the predicted variation carries no real
    signal, and the one-sided p-value on :math:`\beta` is the
    heterogeneity test. Using in-bag predictions here inflates
    :math:`\beta` mechanically.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    cate_predictions : array-like, shape (n,)
        Out-of-bag CATE estimates.
    propensity : array-like, optional
        Treatment probabilities; default the sample mean of D.

    Returns
    -------
    RichResult
        keys: ``alpha`` (calibration), ``beta`` (heterogeneity),
        ``se_beta``, ``p_value`` (one-sided, beta > 0),
        ``heterogeneous`` (p < 0.05), ``n``, ``method``.

    References
    ----------
    Chernozhukov, V., Demirer, M., Duflo, E. & Fernandez-Val, I.
    (2018). Generic machine learning inference on heterogenous
    treatment effects in randomized experiments. arXiv:1712.04802.
    (BLP of the CATE; the calibration/heterogeneity coefficients)
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    tau = np.asarray(cate_predictions, dtype=float).ravel()
    n = y.size
    if not (D.size == n and tau.size == n):
        raise ValueError("y, D, cate_predictions must have equal length.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    ok = np.isfinite(tau)
    if ok.sum() < 10:
        raise ValueError("need at least 10 finite CATE predictions.")
    y, D, tau = y[ok], D[ok], tau[ok]
    m = y.size

    e = np.full(m, D.mean()) if propensity is None else np.asarray(propensity, dtype=float).ravel()[ok]
    if e.size != m:
        raise ValueError("propensity must have one entry per observation.")

    resid_d = D - e
    tau_c = tau - tau.mean()
    X = np.column_stack([np.ones(m), resid_d, resid_d * tau_c])
    b, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    resid = (y - y.mean()) - X @ b
    dof = m - 3
    s2 = float((resid**2).sum() / dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se_beta = float(np.sqrt(cov[2, 2]))
    beta = float(b[2])
    p = float(stats.t.sf(beta / se_beta, dof)) if se_beta > 0 else float("nan")

    return RichResult(
        payload={
            "alpha": float(b[1]),
            "beta": beta,
            "se_beta": se_beta,
            "p_value": p,
            "heterogeneous": bool(p < 0.05),
            "n": int(m),
            "method": "Best-linear-predictor heterogeneity test (Chernozhukov et al. 2018)",
        }
    )


def cheatsheet():
    return "crfhte: regress Y on (D-e) and (D-e)(tau-taubar); beta > 0 = real heterogeneity"
