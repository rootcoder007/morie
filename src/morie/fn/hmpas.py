# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pasting: train base models on samples drawn without replacement."""

from . import _array_core as np

from ._richresult import RichResult
from .hmbag import _stump

__all__ = ["geron_pasting"]


def _lcg_sample(n, k, seed):
    """Draw k distinct indices from [0, n) by a partial Fisher-Yates on an integer LCG."""
    s = int(seed) % 2**32
    pool = np.arange(n)
    for i in range(k):
        s = (1664525 * s + 1013904223) % 2**32
        j = i + (s * (n - i)) // 2**32
        pool[i], pool[j] = pool[j], pool[i]
    return np.sort(pool[:k])


def geron_pasting(X, y, base_estimator=None, n_estimators=10, sample_size=None, seed=0, task="auto"):
    """
    Pasting: train base models on samples drawn WITHOUT replacement.

    Formula: each f_m uses sample of size s drawn without replacement

    Bagging draws with replacement, so a single sample repeats about
    two-thirds of the rows and each model sees a distorted copy of the
    distribution; pasting draws without, so each model sees a clean
    subsample. The trade is diversity: pasting's models are more alike
    (they cannot re-weight rows), which lowers the ensemble's variance
    reduction slightly but also its bias -- Geron's advice is to try
    both, which is why the two functions share an interface.

    The default learner is the least-squares stump DELEGATED from
    :mod:`morie.fn.hmbag`, so the difference between this function and
    bagging is exactly the sampler.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    base_estimator : callable, optional
        ``base_estimator(X_sub, y_sub) -> predict(X) -> array``.
    n_estimators : int, default 10
    sample_size : int or float, optional
        Rows per model; a float is a fraction of n. Default n // 2.
    seed : int, default 0
        Seed of the integer LCG, so runs are byte-identical everywhere.
    task : {"auto", "regression", "classification"}, default "auto"

    Returns
    -------
    result : RichResult
        Keys: predict, train_pred, train_mse, oob_pred, oob_mse,
        samples, estimate, n, method.

    Examples
    --------
    An ensemble of constant learners predicts that constant, so the
    training MSE against y = (1, 3) is exactly 1:

    >>> const = lambda Xb, yb: (lambda A: np.full(np.atleast_2d(np.asarray(A)).shape[0], 2.0))
    >>> r = geron_pasting([[1.0], [2.0]], [1.0, 3.0], const, 4, sample_size=1, seed=3)
    >>> [float(p) for p in r["predict"]([[1.0], [2.0]])]
    [2.0, 2.0]
    >>> float(r["train_mse"])
    1.0

    Samples are drawn without replacement, so no row repeats within one:

    >>> all(len(set(s.tolist())) == len(s) for s in r["samples"])
    True

    Default stumps on a step function stay inside the observed range and
    keep the order:

    >>> r2 = geron_pasting([[1.0], [2.0], [3.0], [4.0]], [1.0, 1.0, 5.0, 5.0],
    ...                    n_estimators=20, sample_size=3, seed=1)
    >>> p = r2["predict"]([[1.0], [4.0]])
    >>> bool(p.min() >= 1.0 and p.max() <= 5.0 and p[0] < p[1])
    True

    References
    ----------
    Geron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"geron_pasting: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = A.shape[0]
    if yv.size != n:
        raise ValueError(f"geron_pasting: X has {n} rows but y has {yv.size} entries")
    M = int(n_estimators)
    if M < 1:
        raise ValueError(f"geron_pasting: n_estimators must be >= 1, got {n_estimators!r}")
    if task not in ("auto", "regression", "classification"):
        raise ValueError(f"geron_pasting: task must be auto, regression or classification, got {task!r}")
    if sample_size is None:
        s = max(1, n // 2)
    elif isinstance(sample_size, float) and 0.0 < sample_size <= 1.0:
        s = max(1, int(round(sample_size * n)))
    else:
        s = int(sample_size)
    if not (1 <= s <= n):
        raise ValueError(f"geron_pasting: sample_size must lie in [1, {n}] (drawn without replacement), got {sample_size!r}")
    classify = task == "classification" or (task == "auto" and set(np.unique(yv).tolist()) <= {0.0, 1.0})

    models, samples = [], []
    stack = np.empty((M, n))
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    for m in range(M):
        idx = _lcg_sample(n, s, seed + 7919 * m)
        f = _stump(A[idx], yv[idx], classify) if base_estimator is None else base_estimator(A[idx], yv[idx])
        if not callable(f):
            raise ValueError("geron_pasting: base_estimator must return a callable predictor")
        pm = np.asarray(f(A), dtype=float).ravel()
        if pm.size != n:
            raise ValueError(f"geron_pasting: estimator {m} returned {pm.size} predictions for {n} rows")
        models.append(f)
        samples.append(idx)
        stack[m] = pm
        oob = np.setdiff1d(np.arange(n), idx)
        oob_sum[oob] += pm[oob]
        oob_cnt[oob] += 1

    def aggregate(P, _c=classify):
        return (P.mean(axis=0) >= 0.5).astype(float) if _c else P.mean(axis=0)

    def predict(Xnew, _models=models, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return aggregate(np.vstack([np.asarray(f(B), dtype=float).ravel() for f in _models]))

    train_pred = aggregate(stack)
    train_mse = float(np.mean((train_pred - yv) ** 2))
    has = oob_cnt > 0
    oob_pred = np.full(n, np.nan)
    oob_pred[has] = oob_sum[has] / oob_cnt[has]
    oob_mse = float(np.mean((oob_pred[has] - yv[has]) ** 2)) if has.any() else float("nan")

    return RichResult(
        title="Pasting",
        summary_lines=[("Estimators", M), ("Sample size", s), ("Training MSE", train_mse)],
        interpretation="Without replacement each model sees an undistorted subsample, at the cost of some diversity.",
        payload={
            "predict": predict,
            "train_pred": train_pred,
            "train_mse": train_mse,
            "oob_pred": oob_pred,
            "oob_mse": oob_mse,
            "samples": samples,
            "estimators": models,
            "sample_size": s,
            "task": "classification" if classify else "regression",
            "estimate": train_mse,
            "n": int(n),
            "method": "Pasting over LCG-seeded samples drawn without replacement",
        },
    )


def cheatsheet():
    return "hmpas: Pasting, base models on samples without replacement"


# compact alias per ledger/NAMING.md
geronpasting = geron_pasting
