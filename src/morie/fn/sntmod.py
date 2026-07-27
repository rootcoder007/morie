# morie.fn -- function file (rootcoder007/morie)
"""Sequential targeted parametric g-formula."""

import numpy as np

from ._richresult import RichResult

__all__ = ["sequential_target_models"]


def sequential_target_models(y, treatment_history, covariate_history, intervention=1.0):
    r"""Iterated conditional expectation (ICE) g-formula.

    Bang and Robins' sequential-regression formulation: set
    :math:`\bar Q_{T+1} = Y`, then for t = T..1 regress
    :math:`\bar Q_{t+1}` on the history :math:`(\bar A_t, \bar L_t)`
    and evaluate the fit with :math:`A_t` set to the regime, giving
    :math:`\bar Q_t`. The estimate is :math:`\hat E[Y(\bar a)] = `
    mean of :math:`\bar Q_1`. Equivalent to the g-formula but needs
    only outcome regressions -- no confounder density models.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    treatment_history : array-like of {0, 1}, shape (n, T) or (n,)
        Observed treatments.
    covariate_history : array-like, shape (n, T) or (n,)
        Time-varying confounder.
    intervention : scalar or array-like shape (T,), default 1
        Static regime.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``regime``, ``Qbar`` (n, the final
        Q-bar_1 values), ``n``, ``n_periods``, ``method``.

    References
    ----------
    Bang, H. & Robins, J. M. (2005). Doubly robust estimation in
    missing data and causal inference models. *Biometrics*, 61(4),
    962-972. (the sequential-regression g-formula; correction 2008,
    *Biometrics* 64(2), 650)
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment_history, dtype=float)
    L = np.asarray(covariate_history, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if L.ndim == 1:
        L = L[:, None]
    n, T = A.shape
    if y.size != n or L.shape != (n, T):
        raise ValueError(f"shapes disagree: y {y.size}, A {A.shape}, L {L.shape}.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("treatment_history must be binary 0/1.")
    regime = np.broadcast_to(np.asarray(intervention, dtype=float).ravel(), (T,)).copy()

    Q = y.copy()
    for t in range(T - 1, -1, -1):
        X = np.column_stack([A[:, : t + 1], L[:, : t + 1]])
        D = np.column_stack([np.ones(n), X])
        b, *_ = np.linalg.lstsq(D, Q, rcond=None)
        Xa = np.column_stack([A[:, :t], np.full(n, regime[t]), L[:, : t + 1]])
        Q = np.column_stack([np.ones(n), Xa]) @ b

    return RichResult(
        payload={
            "estimate": float(Q.mean()),
            "regime": regime,
            "Qbar": Q,
            "n": int(n),
            "n_periods": int(T),
            "method": "Sequential targeted parametric g-formula (ICE)",
        }
    )


def cheatsheet():
    return "sntmod: ICE g-formula -- iterate Qbar_t regressions from T to 1 (Bang-Robins 2005)"
