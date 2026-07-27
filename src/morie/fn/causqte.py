# morie.fn -- function file (rootcoder007/morie)
"""Quantile treatment effect via Firpo IPW."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_quantile_treatment_effect"]


def _weighted_quantile(y, w, tau):
    """Smallest y with normalised cumulative weight >= tau."""
    order = np.argsort(y)
    cw = np.cumsum(w[order])
    cw = cw / cw[-1]
    return float(y[order][np.searchsorted(cw, tau)])


def causal_quantile_treatment_effect(y, T, ps, tau=0.5):
    r"""Firpo's inverse-probability-weighted quantile treatment effect.

    Under selection on observables, the marginal quantiles of the
    potential outcomes are identified by reweighting: :math:`q_1(\tau)`
    solves the :math:`\tau`-quantile problem among the treated with
    weights :math:`1/e(X)`, and :math:`q_0(\tau)` among the controls
    with weights :math:`1/(1-e(X))`. Then

    .. math:: \mathrm{QTE}(\tau) = q_1(\tau) - q_0(\tau).

    Firpo shows the two-step estimator (propensity score, then two
    separate weighted-quantile minimisations) is root-n consistent,
    asymptotically normal, and semiparametrically efficient.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    T : array-like of {0, 1}, shape (n,)
        Treatment indicator.
    ps : array-like, shape (n,)
        Estimated propensity scores, strictly in (0, 1).
    tau : float or array-like in (0, 1), default 0.5
        Quantile level(s).

    Returns
    -------
    RichResult
        keys: ``qte`` (scalar or array matching ``tau``), ``q1``,
        ``q0``, ``tau``, ``n``, ``method``.

    References
    ----------
    Firpo, S. (2007). Efficient semiparametric estimation of quantile
    treatment effects. *Econometrica*, 75(1), 259-276.
    doi:10.1111/j.1468-0262.2007.00738.x.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()
    if not (y.size == T.size == ps.size):
        raise ValueError("y, T, ps must have equal length.")
    if not np.all(np.isin(T, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    if np.any((ps <= 0) | (ps >= 1)):
        raise ValueError("propensity scores must lie strictly in (0, 1).")
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("tau must lie strictly in (0, 1).")
    tr = T == 1
    if tr.sum() == 0 or (~tr).sum() == 0:
        raise ValueError("need both treated and control units.")

    w1 = 1.0 / ps[tr]
    w0 = 1.0 / (1.0 - ps[~tr])
    q1 = np.array([_weighted_quantile(y[tr], w1, t) for t in taus])
    q0 = np.array([_weighted_quantile(y[~tr], w0, t) for t in taus])
    qte = q1 - q0
    scalar = np.isscalar(tau) or np.ndim(tau) == 0

    return RichResult(
        payload={
            "qte": float(qte[0]) if scalar else qte,
            "q1": float(q1[0]) if scalar else q1,
            "q0": float(q0[0]) if scalar else q0,
            "tau": float(taus[0]) if scalar else taus,
            "n": int(y.size),
            "method": "Quantile treatment effect via Firpo IPW",
        }
    )


def cheatsheet():
    return "causqte: QTE(tau) = weighted-quantile difference under 1/e, 1/(1-e) weights (Firpo 2007)"
