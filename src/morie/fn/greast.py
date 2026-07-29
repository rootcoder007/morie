# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early stopping: keep the parameters from the best validation step."""

import numpy as np

from ._richresult import RichResult
from .grmse import geron_linreg_mse_cost
from .grn007 import geron_ch4_mse_gradient_vector

__all__ = ["geron_early_stopping"]

_METHOD = "Early stopping on validation RMSE"


def geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta, theta0=None):
    r"""Train for ``n_iter`` steps, return the parameters from the best one.

    .. math::
        t_{\text{stop}} = \arg\min_t \mathrm{RMSE}_{\text{val}}(t),
        \qquad \text{return } \theta_{t_{\text{stop}}}

    The subtlety worth stating: this is *not* "stop when validation
    error goes up".  Validation error wobbles, so a rule that halts on
    the first uptick halts early and at random.  What is implemented
    here -- and what Géron describes -- is to keep training and roll
    back to the best snapshot, which is why the whole history is
    returned alongside the winner.

    Gradients come from
    :func:`morie.fn.grn007.geron_ch4_mse_gradient_vector`, errors from
    :func:`morie.fn.grmse.geron_linreg_mse_cost`.

    Parameters
    ----------
    X_train : array-like, shape (m, n)
    y_train : array-like, shape (m,)
    X_val : array-like, shape (m_val, n)
    y_val : array-like, shape (m_val,)
    n_iter : int
        Batch gradient steps, at least 1.
    eta : float
        Positive learning rate.
    theta0 : array-like, optional
        Starting parameters, default zeros.

    Returns
    -------
    RichResult
        Payload keys ``theta`` (best), ``best_iteration``,
        ``best_val_rmse``, ``val_rmse_history``, ``train_rmse_history``
        (both length ``n_iter + 1``, index 0 = before any step),
        ``final_val_rmse``, ``overfitting_detected``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Early Stopping section.

    Examples
    --------
    Clean linear data: validation error falls the whole way, so the best
    snapshot is the last one and nothing is rolled back.

    >>> Xtr = [[1.0, float(i)] for i in range(6)]
    >>> ytr = [float(i) for i in range(6)]
    >>> Xva = [[1.0, 10.0], [1.0, 11.0]]
    >>> r = geron_early_stopping(Xtr, ytr, Xva, [10.0, 11.0], n_iter=50, eta=0.02)
    >>> r["best_iteration"]
    50
    >>> r["best_val_rmse"] < r["val_rmse_history"][0]
    True

    Give the training set a systematic offset the validation set does
    not share, and the run turns: the best snapshot is an interior step
    and the final model is worse than it.

    >>> ytr2 = [float(i) + 5.0 for i in range(6)]
    >>> r2 = geron_early_stopping(Xtr, ytr2, Xva, [10.0, 11.0], n_iter=200, eta=0.02)
    >>> 0 < r2["best_iteration"] < 200
    True
    >>> r2["overfitting_detected"]
    True
    """
    A = np.atleast_2d(np.asarray(X_train, dtype=float))
    ytr = np.asarray(y_train, dtype=float).ravel()
    V = np.atleast_2d(np.asarray(X_val, dtype=float))
    yva = np.asarray(y_val, dtype=float).ravel()
    if A.ndim != 2 or V.ndim != 2:
        raise ValueError(f"X_train and X_val must be 2-D, got {A.shape} and {V.shape}.")
    if A.shape[1] != V.shape[1]:
        raise ValueError(
            f"X_train has {A.shape[1]} columns but X_val has {V.shape[1]}."
        )
    if ytr.size != A.shape[0]:
        raise ValueError(f"y_train has {ytr.size} entries but X_train has {A.shape[0]} rows.")
    if yva.size != V.shape[0]:
        raise ValueError(f"y_val has {yva.size} entries but X_val has {V.shape[0]} rows.")
    if V.shape[0] == 0:
        raise ValueError("X_val is empty; early stopping needs a validation set.")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    th = np.zeros(A.shape[1]) if theta0 is None else np.asarray(theta0, dtype=float).ravel()
    if th.size != A.shape[1]:
        raise ValueError(f"theta0 has {th.size} entries but X_train has {A.shape[1]} columns.")

    tr_hist = [geron_linreg_mse_cost(A, ytr, th)["rmse"]]
    va_hist = [geron_linreg_mse_cost(V, yva, th)["rmse"]]
    best_theta, best_it = th.copy(), 0

    for it in range(1, n_iter + 1):
        g = geron_ch4_mse_gradient_vector(A, ytr, th)["gradient"]
        th = th - eta * np.asarray(g, dtype=float)
        if not np.all(np.isfinite(th)):
            raise ValueError(
                f"parameters diverged to non-finite values at step {it}; "
                f"eta = {eta} is too large."
            )
        tr_hist.append(geron_linreg_mse_cost(A, ytr, th)["rmse"])
        v = geron_linreg_mse_cost(V, yva, th)["rmse"]
        va_hist.append(v)
        if v < va_hist[best_it]:
            best_theta, best_it = th.copy(), it

    return RichResult(
        title="Early stopping",
        summary_lines=[("Best iteration", best_it),
                       ("Best val RMSE", va_hist[best_it]),
                       ("Final val RMSE", va_hist[-1])],
        payload={
            "theta": best_theta.tolist(),
            "best_iteration": int(best_it),
            "best_val_rmse": float(va_hist[best_it]),
            "final_val_rmse": float(va_hist[-1]),
            "val_rmse_history": va_hist,
            "train_rmse_history": tr_hist,
            "overfitting_detected": bool(best_it < n_iter),
            "estimate": best_theta.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "greast: run all n_iter steps, roll back to argmin val RMSE (not first uptick)"
