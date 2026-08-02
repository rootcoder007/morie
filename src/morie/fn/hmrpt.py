# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Random patches: subsample rows and features per base model."""

from . import _array_core as np

from ._richresult import RichResult
from .hmbag import _stump
from .hmpas import _lcg_sample
from .hmrsp import _lcg_features

__all__ = ["geron_random_patches"]


def geron_random_patches(X, y, base_estimator=None, n_estimators=10, max_samples=None,
                         max_features=None, seed=0, task="auto", bootstrap=False):
    """
    Random patches: subsample BOTH rows and features per base model.

    Formula: each f_m uses random rows I_m and features S_m

    Sampling both axes is the strongest decorrelation of the bagging
    family and the cheapest per model, which is the point: on a wide,
    tall dataset each member touches a small rectangle of the data and
    the ensemble still covers all of it. The extra randomness raises
    each member's bias, so patches want MORE estimators than plain
    bagging, not fewer.

    Rows are drawn without replacement by default (pasting-style); set
    ``bootstrap=True`` for with-replacement draws. The default learner is
    the least-squares stump DELEGATED from :mod:`morie.fn.hmbag`, and the
    two samplers are shared with :mod:`morie.fn.hmpas` and
    :mod:`morie.fn.hmrsp`.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    base_estimator : callable, optional
        ``base_estimator(X_patch, y_patch) -> predict(X_patch) -> array``.
    n_estimators : int, default 10
    max_samples : int or float, optional
        Rows per model; default n // 2.
    max_features : int or float, optional
        Columns per model; default ceil(sqrt(d)).
    seed : int, default 0
    task : {"auto", "regression", "classification"}, default "auto"
    bootstrap : bool, default False
        Draw rows with replacement instead.

    Returns
    -------
    result : RichResult
        Keys: predict, train_pred, train_mse, patches, feature_usage,
        row_usage, estimate, n, method.

    Examples
    --------
    >>> const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 2.0))
    >>> X = [[1.0, 0.0, 5.0], [2.0, 1.0, 4.0], [3.0, 2.0, 3.0], [4.0, 3.0, 2.0]]
    >>> r = geron_random_patches(X, [1.0, 1.0, 3.0, 3.0], const, 8,
    ...                          max_samples=2, max_features=2, seed=11)
    >>> [float(p) for p in r["predict"](X)]
    [2.0, 2.0, 2.0, 2.0]

    Every patch is exactly 2 rows by 2 columns:

    >>> sorted({(len(i), len(f)) for i, f in r["patches"]})
    [(2, 2)]

    Training MSE against y = (1, 1, 3, 3) for a constant 2 is 1:

    >>> float(r["train_mse"])
    1.0

    References
    ----------
    Geron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_random_patches: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n, d = A.shape
    if yv.size != n:
        raise ValueError(f"geron_random_patches: X has {n} rows but y has {yv.size} entries")
    M = int(n_estimators)
    if M < 1:
        raise ValueError(f"geron_random_patches: n_estimators must be >= 1, got {n_estimators!r}")
    if task not in ("auto", "regression", "classification"):
        raise ValueError(f"geron_random_patches: task must be auto, regression or classification, got {task!r}")
    if max_samples is None:
        s = max(1, n // 2)
    elif isinstance(max_samples, float) and 0.0 < max_samples <= 1.0:
        s = max(1, int(round(max_samples * n)))
    else:
        s = int(max_samples)
    if not (1 <= s <= n) and not bootstrap:
        raise ValueError(f"geron_random_patches: max_samples must lie in [1, {n}] without replacement, got {max_samples!r}")
    if s < 1:
        raise ValueError(f"geron_random_patches: max_samples must be >= 1, got {max_samples!r}")
    if max_features is None:
        k = int(np.ceil(np.sqrt(d)))
    elif isinstance(max_features, float) and 0.0 < max_features <= 1.0:
        k = max(1, int(round(max_features * d)))
    else:
        k = int(max_features)
    if not (1 <= k <= d):
        raise ValueError(f"geron_random_patches: max_features must lie in [1, {d}], got {max_features!r}")
    classify = task == "classification" or (task == "auto" and set(np.unique(yv).tolist()) <= {0.0, 1.0})

    models, patches = [], []
    stack = np.empty((M, n))
    fuse = np.zeros(d, dtype=int)
    ruse = np.zeros(n, dtype=int)
    for m in range(M):
        if bootstrap:
            st = (seed + 7919 * m) % 2**32
            rows = np.empty(s, dtype=int)
            for i in range(s):
                st = (1664525 * st + 1013904223) % 2**32
                rows[i] = (st * n) // 2**32
        else:
            rows = _lcg_sample(n, s, seed + 7919 * m)
        cols = _lcg_features(d, k, seed + 104729 * m + 13)
        Xp = A[np.ix_(rows, cols)]
        f = _stump(Xp, yv[rows], classify) if base_estimator is None else base_estimator(Xp, yv[rows])
        if not callable(f):
            raise ValueError("geron_random_patches: base_estimator must return a callable predictor")
        pm = np.asarray(f(A[:, cols]), dtype=float).ravel()
        if pm.size != n:
            raise ValueError(f"geron_random_patches: estimator {m} returned {pm.size} predictions for {n} rows")
        models.append((f, cols))
        patches.append((rows, cols))
        fuse[cols] += 1
        ruse[np.unique(rows)] += 1
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
        title="Random patches",
        summary_lines=[("Estimators", M), ("Patch", (s, k)), ("Training MSE", train_mse)],
        interpretation="Sampling both axes maximises diversity and per-member bias; budget more estimators for it.",
        payload={
            "predict": predict,
            "train_pred": train_pred,
            "train_mse": train_mse,
            "patches": patches,
            "feature_usage": fuse,
            "row_usage": ruse,
            "estimators": models,
            "max_samples": s,
            "max_features": k,
            "task": "classification" if classify else "regression",
            "estimate": train_mse,
            "n": int(n),
            "method": "Random patches: LCG-drawn row and column subsets per estimator",
        },
    )


def cheatsheet():
    return "hmrpt: Random patches, subsampling rows and features per model"
