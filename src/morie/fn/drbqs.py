# morie.fn -- function file (rootcoder007/morie)
"""DR-DiD quantile treatment effect."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["dr_did_quantile"]


def _weighted_quantile(v, w, tau):
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw / cw[-1], tau)])


def dr_did_quantile(y_pre, y_post, D, X, quantile=0.5):
    r"""Quantile treatment effect on the treated for panel DiD.

    Works on the outcome *changes* :math:`\Delta Y = Y_{post} -
    Y_{pre}` (the panel analogue of the distributional DiD
    assumption): the QTT at level :math:`\tau` is the treated
    :math:`\tau`-quantile of :math:`\Delta Y` minus the control
    :math:`\tau`-quantile after reweighting controls to the treated
    covariate distribution with ATT weights
    :math:`\hat e(X)/(1-\hat e(X))`.

    Parameters
    ----------
    y_pre, y_post : array-like, shape (n,)
        Panel outcomes.
    D : array-like of {0, 1}, shape (n,)
        Treatment group.
    X : array-like, shape (n,) or (n, p)
        Covariates for the propensity model.
    quantile : float or array-like in (0, 1), default 0.5
        Quantile level(s).

    Returns
    -------
    RichResult
        keys: ``qtt``, ``q_treated``, ``q_control``, ``tau``, ``n``,
        ``method``.

    References
    ----------
    Callaway, B. & Li, T. (2019). Quantile treatment effects in
    difference in differences models with panel data. *Quantitative
    Economics*, 10(4), 1579-1618. doi:10.3982/QE935. (QTT for panel
    DiD; here the change-in-outcomes variant with covariate
    reweighting)

    Firpo, S. (2007). Efficient semiparametric estimation of quantile
    treatment effects. *Econometrica*, 75(1), 259-276. (the IPW
    weighted-quantile step)
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    y_post = np.asarray(y_post, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = y_pre.size
    if not (y_post.size == n and D.size == n and X.shape[0] == n):
        raise ValueError("y_pre, y_post, D, X must share their first dimension.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    if D.sum() == 0 or D.sum() == n:
        raise ValueError("need both treated and control units.")
    taus = np.atleast_1d(np.asarray(quantile, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantile must lie strictly in (0, 1).")

    dy = y_post - y_pre
    e = np.clip(_logit_fit(X, D), 0.01, 0.99)
    tr = D == 1
    w_c = e[~tr] / (1 - e[~tr])  # reweight controls to the treated

    q1 = np.array([_weighted_quantile(dy[tr], np.ones(int(tr.sum())), t) for t in taus])
    q0 = np.array([_weighted_quantile(dy[~tr], w_c, t) for t in taus])
    qtt = q1 - q0
    scalar = np.ndim(quantile) == 0

    return RichResult(
        payload={
            "qtt": float(qtt[0]) if scalar else qtt,
            "q_treated": float(q1[0]) if scalar else q1,
            "q_control": float(q0[0]) if scalar else q0,
            "tau": float(taus[0]) if scalar else taus,
            "n": int(n),
            "method": "DR-DiD quantile treatment effect (change distribution, ATT weights)",
        }
    )


def cheatsheet():
    return "drbqs: QTT(tau) on Delta-Y with e/(1-e)-reweighted controls"
