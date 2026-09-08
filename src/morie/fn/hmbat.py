# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch (offline) learning: train once on full dataset, then deploy."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_batch_learning"]


def geron_batch_learning(X, y, fit_intercept=False, ridge=0.0):
    """
    Batch (offline) learning: train once on the full dataset, then deploy.

    Formula: theta = argmin_theta L(theta; D_train)

    The batch loss is the (optionally ridge-penalised) mean squared error,
    minimised in closed form. The fitted parameters are frozen: the returned
    `predict` closure applies them to new data without further updating,
    which is exactly what makes the regime "offline".

    Parameters
    ----------
    X : array-like, shape (m, k)
        Training design matrix.
    y : array-like, shape (m,)
        Training targets.
    fit_intercept : bool, default False
        Prepend a column of ones before solving.
    ridge : float, default 0.0
        L2 penalty; must be non-negative. The intercept is left unpenalised.

    Returns
    -------
    result : RichResult
        Keys: theta, predict, train_mse, r2, estimate, n, method.

    Examples
    --------
    >>> r = geron_batch_learning([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [1.0, 3.0, 5.0])
    >>> [round(float(t), 10) for t in r["theta"]]
    [1.0, 2.0]
    >>> round(float(r["train_mse"]), 12)
    0.0
    >>> [round(float(p), 10) for p in r["predict"]([[1.0, 3.0]])]
    [7.0]

    References
    ----------
    Géron Ch 1
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    if Xm.ndim != 2:
        raise ValueError(f"geron_batch_learning: X must be 2-D, got ndim={Xm.ndim}")
    yv = np.asarray(y, dtype=float).ravel()
    if Xm.shape[0] == 0:
        raise ValueError("geron_batch_learning: training set is empty")
    if yv.size != Xm.shape[0]:
        raise ValueError(f"geron_batch_learning: X has {Xm.shape[0]} rows but y has {yv.size} entries")
    if not (np.all(np.isfinite(Xm)) and np.all(np.isfinite(yv))):
        raise ValueError("geron_batch_learning: X and y must be finite")
    lam = float(ridge)
    if lam < 0:
        raise ValueError("geron_batch_learning: ridge must be non-negative")

    D = np.hstack([np.ones((Xm.shape[0], 1)), Xm]) if fit_intercept else Xm
    k = D.shape[1]
    if lam == 0.0:
        theta, *_ = np.linalg.lstsq(D, yv, rcond=None)
    else:
        P = np.eye(k) * lam
        if fit_intercept:
            P[0, 0] = 0.0
        theta = np.linalg.solve(D.T @ D + P, D.T @ yv)

    fitted = D @ theta
    resid = yv - fitted
    train_mse = float(np.mean(resid**2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / tss if tss > 0 else float("nan")

    def predict(Xnew, _theta=theta, _fi=fit_intercept, _k=Xm.shape[1]):
        A = np.asarray(Xnew, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.shape[1] != _k:
            raise ValueError(f"predict: expected {_k} features, got {A.shape[1]}")
        if _fi:
            A = np.hstack([np.ones((A.shape[0], 1)), A])
        return A @ _theta

    return RichResult(
        title="Batch (offline) learning",
        summary_lines=[("Training MSE", train_mse), ("R^2", r2), ("Parameters", k)],
        interpretation="Parameters are frozen after this single pass; new data requires a full retrain.",
        payload={
            "theta": theta,
            "predict": predict,
            "fitted": fitted,
            "residuals": resid,
            "train_mse": train_mse,
            "r2": r2,
            "estimate": train_mse,
            "n": int(Xm.shape[0]),
            "method": "Batch learning: closed-form minimisation of the full-dataset squared-error loss",
        },
    )


def cheatsheet():
    return "hmbat: Batch (offline) learning: train once on full dataset, then deploy"
