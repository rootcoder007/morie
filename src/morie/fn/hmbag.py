# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bagging"]


def _lcg_indices(n, count, seed):
    """Exact-integer LCG draw of `count` indices in [0, n); reproducible everywhere."""
    s = int(seed) % 2**32
    out = np.empty(count, dtype=int)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s * n) // 2**32
    return out


def _stump(Xb, yb, classify):
    """Least-squares (or majority-vote) 1-split stump on a bootstrap sample."""
    n, d = Xb.shape
    best = (np.inf, 0, np.inf, float(np.mean(yb)), float(np.mean(yb)))
    for j in range(d):
        vals = np.unique(Xb[:, j])
        if vals.size < 2:
            continue
        for thr in (vals[:-1] + vals[1:]) / 2.0:
            left = Xb[:, j] <= thr
            if not left.any() or left.all():
                continue
            lv, rv = yb[left], yb[~left]
            lp, rp = float(np.mean(lv)), float(np.mean(rv))
            sse = float(np.sum((lv - lp) ** 2) + np.sum((rv - rp) ** 2))
            if sse < best[0]:
                best = (sse, j, float(thr), lp, rp)
    _, j, thr, lp, rp = best
    if classify:
        lp = 1.0 if lp >= 0.5 else 0.0
        rp = 1.0 if rp >= 0.5 else 0.0

    def predict(A, _j=j, _t=thr, _l=lp, _r=rp):
        B = np.atleast_2d(np.asarray(A, dtype=float))
        return np.where(B[:, _j] <= _t, _l, _r).astype(float)

    return predict


def geron_bagging(X, y, base_estimator=None, n_estimators=10, seed=0, task="auto"):
    """
    Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs.

    Formula: y_hat = (1/M) sum_m f_m(x), f_m trained on bootstrap of D

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    base_estimator : callable, optional
        ``base_estimator(X_boot, y_boot) -> predict(X) -> array``.
        Default is a least-squares decision stump.
    n_estimators : int
        Number of bootstrap replicates (>= 1).
    seed : int
        Seed of the integer LCG used to draw the bootstrap indices, so runs
        are byte-identical across machines.
    task : {"auto", "regression", "classification"}
        Aggregation rule: mean for regression, majority vote for
        classification. "auto" picks classification when y takes exactly the
        two values {0, 1}.

    Returns
    -------
    result : RichResult
        Keys: predict, train_pred, train_mse, oob_pred, oob_mse, estimators,
        estimate, n, method.

    Examples
    --------
    Aggregation is a plain average, so an ensemble of constant learners
    predicts that constant and its training MSE is exact:

    >>> const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 2.0))
    >>> r = geron_bagging([[1.0], [2.0]], [1.0, 3.0], const, 4, seed=7)
    >>> [float(p) for p in r["predict"]([[1.0], [2.0]])]
    [2.0, 2.0]
    >>> float(r["train_mse"])
    1.0

    Default stumps on a step function stay inside the observed range:

    >>> r2 = geron_bagging([[1.0], [2.0], [3.0], [4.0]], [1.0, 1.0, 5.0, 5.0], n_estimators=25, seed=1)
    >>> p = r2["predict"]([[1.0], [4.0]])
    >>> bool(p.min() >= 1.0 and p.max() <= 5.0)
    True
    >>> bool(p[0] < p[1])
    True

    References
    ----------
    Géron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_bagging: X must be 2-D, got ndim={A.ndim}")
    n = A.shape[0]
    if n == 0:
        raise ValueError("geron_bagging: X has no rows")
    yv = np.asarray(y, dtype=float).ravel()
    if yv.size != n:
        raise ValueError(f"geron_bagging: X has {n} rows but y has {yv.size} entries")
    M = int(n_estimators)
    if M < 1:
        raise ValueError("geron_bagging: n_estimators must be >= 1")
    if task not in ("auto", "regression", "classification"):
        raise ValueError(f"geron_bagging: task must be auto, regression or classification, got {task!r}")
    classify = task == "classification" or (task == "auto" and set(np.unique(yv).tolist()) <= {0.0, 1.0})

    models = []
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    train_stack = np.empty((M, n))
    for m in range(M):
        idx = _lcg_indices(n, n, seed + 7919 * m)
        Xb, yb = A[idx], yv[idx]
        f = _stump(Xb, yb, classify) if base_estimator is None else base_estimator(Xb, yb)
        if not callable(f):
            raise ValueError("geron_bagging: base_estimator must return a callable predictor")
        pm = np.asarray(f(A), dtype=float).ravel()
        if pm.size != n:
            raise ValueError(f"geron_bagging: estimator {m} returned {pm.size} predictions for {n} rows")
        models.append(f)
        train_stack[m] = pm
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        oob_sum[oob] += pm[oob]
        oob_cnt[oob] += 1

    def aggregate(P, _classify=classify):
        if _classify:
            return (P.mean(axis=0) >= 0.5).astype(float)
        return P.mean(axis=0)

    def predict(Xnew, _models=models, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        P = np.vstack([np.asarray(f(B), dtype=float).ravel() for f in _models])
        return aggregate(P)

    train_pred = aggregate(train_stack)
    train_mse = float(np.mean((train_pred - yv) ** 2))
    has_oob = oob_cnt > 0
    oob_pred = np.full(n, np.nan)
    oob_pred[has_oob] = oob_sum[has_oob] / oob_cnt[has_oob]
    oob_mse = float(np.mean((oob_pred[has_oob] - yv[has_oob]) ** 2)) if np.any(has_oob) else float("nan")

    return RichResult(
        title="Bagging (bootstrap aggregating)",
        summary_lines=[("Estimators", M), ("Training MSE", train_mse), ("OOB MSE", oob_mse)],
        payload={
            "predict": predict,
            "train_pred": train_pred,
            "train_mse": train_mse,
            "oob_pred": oob_pred,
            "oob_mse": oob_mse,
            "oob_coverage": float(np.mean(has_oob)),
            "estimators": models,
            "member_preds": train_stack,
            "task": "classification" if classify else "regression",
            "estimate": train_mse,
            "n": int(n),
            "method": "Bagging over LCG-seeded bootstrap replicates",
        },
    )


def cheatsheet():
    return "hmbag: Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs"
