# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early stopping: halt training when validation error stops decreasing."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_early_stopping"]


def geron_early_stopping(X_train, y_train, X_val, y_val, n_iter=100, eta=0.01, patience=None, fit_intercept=True):
    """
    Early stopping: halt training when validation error stops decreasing.

    Formula: t_stop = argmin_t RMSE_val(t); return theta_{t_stop}

    Batch gradient descent is actually run on a linear model, and both the
    training and validation RMSE are recorded at every iteration. The best
    parameters are *kept*, not merely reported: ``theta`` is the snapshot
    at ``best_iter``, which is what makes this a regulariser rather than a
    stopping heuristic.

    Two stopping rules are computed. ``best_iter`` is the global argmin
    over the whole run -- available only in hindsight. With ``patience``
    set, ``stopped_iter`` is where an online rule would actually have
    halted, after ``patience`` iterations with no improvement, and the
    difference between the two is exactly the cost of not being able to
    see the future.

    Training RMSE is monotone for gradient descent on a convex loss with a
    small enough step; validation RMSE is U-shaped when the model
    overfits, and ``is_u_shaped`` says whether that happened here.

    Parameters
    ----------
    X_train, X_val : array-like, shape (m, n)
    y_train, y_val : array-like
    n_iter : int, default 100
    eta : float, default 0.01
        Learning rate.
    patience : int, optional
        Iterations without improvement before an online rule stops.
    fit_intercept : bool, default True

    Returns
    -------
    result : RichResult
        Keys: theta, best_iter, best_val_rmse, stopped_iter, val_rmse,
        train_rmse, is_u_shaped, final_val_rmse, estimate, n, method.

    Examples
    --------
    A clean linear problem: validation error falls and the best iteration
    is the last one, so early stopping does nothing.

    >>> Xt = [[0.0], [1.0], [2.0], [3.0]]
    >>> yt = [0.0, 2.0, 4.0, 6.0]
    >>> r = geron_early_stopping(Xt, yt, [[4.0], [5.0]], [8.0, 10.0], n_iter=200, eta=0.05)
    >>> r["best_iter"] == 200
    True
    >>> r["best_val_rmse"] < 0.1
    True
    >>> round(r["theta"][1], 2)
    2.0

    A validation set that disagrees with training makes the curve
    U-shaped, and the best snapshot is not the last:

    >>> r2 = geron_early_stopping(Xt, yt, [[0.0], [1.0]], [3.0, 3.0], n_iter=200, eta=0.05)
    >>> r2["is_u_shaped"]
    True
    >>> r2["best_iter"] < 200
    True
    >>> r2["best_val_rmse"] <= r2["final_val_rmse"]
    True

    With patience, the online rule stops after the first stretch of no
    improvement:

    >>> r3 = geron_early_stopping(Xt, yt, [[0.0], [1.0]], [3.0, 3.0], n_iter=200,
    ...                           eta=0.05, patience=5)
    >>> r3["stopped_iter"] == r3["best_iter"] + 5
    True

    References
    ----------
    Géron Ch 4
    """
    Xt = np.atleast_2d(np.asarray(X_train, dtype=float))
    yt = np.asarray(y_train, dtype=float).ravel()
    Xv = np.atleast_2d(np.asarray(X_val, dtype=float))
    yv = np.asarray(y_val, dtype=float).ravel()
    for name, A, b in (("train", Xt, yt), ("validation", Xv, yv)):
        if A.size == 0 or b.size == 0:
            raise ValueError(f"geron_early_stopping: the {name} set is empty")
        if A.shape[0] != b.size:
            raise ValueError(f"geron_early_stopping: the {name} set has {A.shape[0]} rows but {b.size} targets")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError(f"geron_early_stopping: the {name} set contains non-finite values")
    if Xt.shape[1] != Xv.shape[1]:
        raise ValueError(f"geron_early_stopping: train has {Xt.shape[1]} features but validation has {Xv.shape[1]}")
    T = int(n_iter)
    if T < 1:
        raise ValueError(f"geron_early_stopping: n_iter must be >= 1, got {n_iter!r}")
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_early_stopping: eta must be positive and finite, got {eta!r}")
    pat = None if patience is None else int(patience)
    if pat is not None and pat < 1:
        raise ValueError(f"geron_early_stopping: patience must be >= 1, got {patience!r}")

    A = np.hstack([np.ones((Xt.shape[0], 1)), Xt]) if fit_intercept else Xt
    B = np.hstack([np.ones((Xv.shape[0], 1)), Xv]) if fit_intercept else Xv
    theta = np.zeros(A.shape[1])
    m = A.shape[0]

    def rmse(D, t, target):
        return float(np.sqrt(np.mean((D @ t - target) ** 2)))

    tr_hist = [rmse(A, theta, yt)]
    va_hist = [rmse(B, theta, yv)]
    best = (va_hist[0], 0, theta.copy())
    stopped = None
    since = 0
    for it in range(1, T + 1):
        grad = (2.0 / m) * (A.T @ (A @ theta - yt))
        theta = theta - lr * grad
        if not np.all(np.isfinite(theta)):
            raise ValueError(f"geron_early_stopping: parameters diverged at iteration {it}; eta={lr} is too large")
        tr_hist.append(rmse(A, theta, yt))
        v = rmse(B, theta, yv)
        va_hist.append(v)
        if v < best[0] - 1e-15:
            best = (v, it, theta.copy())
            since = 0
        else:
            since += 1
            if pat is not None and since >= pat and stopped is None:
                stopped = it

    if pat is not None and stopped is None:
        stopped = T
    u_shaped = bool(best[1] < T)

    return RichResult(
        title="Early stopping",
        summary_lines=[("Best iteration", best[1]), ("Best val RMSE", best[0]), ("Final val RMSE", va_hist[-1])],
        interpretation="Keeping the best snapshot is what makes early stopping a regulariser rather than a shortcut.",
        payload={
            "theta": best[2].tolist(),
            "best_iter": int(best[1]),
            "best_val_rmse": float(best[0]),
            "stopped_iter": stopped,
            "val_rmse": va_hist,
            "train_rmse": tr_hist,
            "final_theta": theta.tolist(),
            "final_val_rmse": float(va_hist[-1]),
            "is_u_shaped": u_shaped,
            "patience": pat,
            "eta": lr,
            "estimate": float(best[0]),
            "n": int(m),
            "method": "batch gradient descent with best-snapshot early stopping",
        },
    )


def cheatsheet():
    return "hmearl: Early stopping: halt training when validation error stops decreasing"
