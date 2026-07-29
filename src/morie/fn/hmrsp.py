# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Random subspaces: bag the features, keep every row."""

import numpy as np

from ._richresult import RichResult
from .hmbag import _stump

__all__ = ["geron_random_subspaces"]


def _lcg_features(d, k, seed):
    s = int(seed) % 2**32
    pool = np.arange(d)
    for i in range(k):
        s = (1664525 * s + 1013904223) % 2**32
        j = i + (s * (d - i)) // 2**32
        pool[i], pool[j] = pool[j], pool[i]
    return np.sort(pool[:k])


def geron_random_subspaces(X, y, base_estimator=None, n_estimators=10, max_features=None, seed=0, task="auto"):
    """
    Random subspaces: bag features per model without subsampling rows.

    Formula: each f_m uses random subset of features S_m

    Every model trains on all the data but through a different keyhole.
    That decorrelates the members when the columns are correlated -- if
    two features carry the same signal, no single model can lean on both
    -- which is exactly the case where row bagging alone barely helps,
    because every bootstrap still contains the same dominant feature.
    High-dimensional inputs (images, spectra) are where this pays.

    The default learner is the least-squares stump DELEGATED from
    :mod:`morie.fn.hmbag`.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    base_estimator : callable, optional
        ``base_estimator(X_sub, y_sub) -> predict(X_sub) -> array``,
        where ``X_sub`` already has only the model's own columns.
    n_estimators : int, default 10
    max_features : int or float, optional
        Columns per model; a float is a fraction of d. Default
        ceil(sqrt(d)).
    seed : int, default 0
    task : {"auto", "regression", "classification"}, default "auto"

    Returns
    -------
    result : RichResult
        Keys: predict, train_pred, train_mse, feature_sets,
        feature_usage, estimate, n, method.

    Examples
    --------
    >>> const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 2.0))
    >>> X = [[1.0, 0.0, 5.0, 9.0], [2.0, 1.0, 4.0, 8.0]]
    >>> r = geron_random_subspaces(X, [1.0, 3.0], const, 6, max_features=2, seed=5)
    >>> [float(p) for p in r["predict"](X)]
    [2.0, 2.0]
    >>> sorted({len(s) for s in r["feature_sets"]})
    [2]

    Each model sees only its own columns, and every column gets used at
    least once over enough models:

    >>> bool((r["feature_usage"] > 0).all())
    True

    References
    ----------
    Geron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_random_subspaces: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n, d = A.shape
    if yv.size != n:
        raise ValueError(f"geron_random_subspaces: X has {n} rows but y has {yv.size} entries")
    M = int(n_estimators)
    if M < 1:
        raise ValueError(f"geron_random_subspaces: n_estimators must be >= 1, got {n_estimators!r}")
    if task not in ("auto", "regression", "classification"):
        raise ValueError(f"geron_random_subspaces: task must be auto, regression or classification, got {task!r}")
    if max_features is None:
        k = int(np.ceil(np.sqrt(d)))
    elif isinstance(max_features, float) and 0.0 < max_features <= 1.0:
        k = max(1, int(round(max_features * d)))
    else:
        k = int(max_features)
    if not (1 <= k <= d):
        raise ValueError(f"geron_random_subspaces: max_features must lie in [1, {d}], got {max_features!r}")
    classify = task == "classification" or (task == "auto" and set(np.unique(yv).tolist()) <= {0.0, 1.0})

    models, sets = [], []
    stack = np.empty((M, n))
    usage = np.zeros(d, dtype=int)
    for m in range(M):
        cols = _lcg_features(d, k, seed + 7919 * m)
        f = _stump(A[:, cols], yv, classify) if base_estimator is None else base_estimator(A[:, cols], yv)
        if not callable(f):
            raise ValueError("geron_random_subspaces: base_estimator must return a callable predictor")
        pm = np.asarray(f(A[:, cols]), dtype=float).ravel()
        if pm.size != n:
            raise ValueError(f"geron_random_subspaces: estimator {m} returned {pm.size} predictions for {n} rows")
        models.append((f, cols))
        sets.append(cols)
        usage[cols] += 1
        stack[m] = pm

    def aggregate(P, _c=classify):
        return (P.mean(axis=0) >= 0.5).astype(float) if _c else P.mean(axis=0)

    def predict(Xnew, _models=models, _d=d):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return aggregate(np.vstack([np.asarray(f(B[:, c]), dtype=float).ravel() for f, c in _models]))

    train_pred = aggregate(stack)
    train_mse = float(np.mean((train_pred - yv) ** 2))
    return RichResult(
        title="Random subspaces",
        summary_lines=[("Estimators", M), ("Features per model", k), ("Training MSE", train_mse)],
        interpretation="Feature bagging decorrelates members when columns are redundant, where row bagging cannot.",
        payload={
            "predict": predict,
            "train_pred": train_pred,
            "train_mse": train_mse,
            "feature_sets": sets,
            "feature_usage": usage,
            "estimators": models,
            "max_features": k,
            "task": "classification" if classify else "regression",
            "estimate": train_mse,
            "n": int(n),
            "method": "Random subspaces: all rows, an LCG-drawn column subset per model",
        },
    )


def cheatsheet():
    return "hmrsp: Random subspaces, feature bagging without row subsampling"
