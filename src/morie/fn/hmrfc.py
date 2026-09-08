# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Random forest: bagged trees with random feature splits."""

from . import _array_core as np

from ._richresult import RichResult
from .hmrdt import _grow, _predict_tree

__all__ = ["geron_random_forest"]


def _lcg_bootstrap(n, seed):
    s = int(seed) % 2**32
    out = np.empty(n, dtype=int)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s * n) // 2**32
    return out


def _lcg_cols(d, k, seed):
    s = int(seed) % 2**32
    pool = np.arange(d)
    for i in range(k):
        s = (1664525 * s + 1013904223) % 2**32
        j = i + (s * (d - i)) // 2**32
        pool[i], pool[j] = pool[j], pool[i]
    return np.sort(pool[:k])


def geron_random_forest(X, y, n_estimators=10, max_features="sqrt", seed=0, max_depth=4,
                        min_samples_leaf=1, task="auto"):
    """
    Random forest: bagging of decision trees with random feature splits.

    Formula: y_hat = majority vote of M random-split trees

    Bagging alone leaves the trees correlated -- one dominant feature is
    chosen at the root of nearly every bootstrap, so the ensemble
    averages M copies of the same mistake. Restricting each SPLIT to a
    random subset of columns breaks that: individual trees get worse,
    the ensemble gets better. That trade is the whole idea.

    The trees are grown by the CART routine shared with
    :mod:`morie.fn.hmrdt` (whose split cost is in turn delegated to
    :func:`morie.fn.grcart.geron_cart_split_cost`); the only change here
    is the per-node column sample and the bootstrap.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    n_estimators : int, default 10
    max_features : {"sqrt", "log2", "all"}, int or float, default "sqrt"
        Columns considered at each split.
    seed : int, default 0
        Integer-LCG seed, so the forest is reproducible everywhere.
    max_depth : int, default 4
    min_samples_leaf : int, default 1
    task : {"auto", "classification", "regression"}, default "auto"

    Returns
    -------
    result : RichResult
        Keys: predict, predictions, accuracy or mse, oob_score, trees,
        feature_importance, estimate, n, method.

    Examples
    --------
    A separable two-class problem is fit exactly:

    >>> X = [[1.0, 9.0], [2.0, 8.0], [8.0, 2.0], [9.0, 1.0]]
    >>> r = geron_random_forest(X, [0, 0, 1, 1], n_estimators=9, seed=4)
    >>> float(r["accuracy"])
    1.0
    >>> [int(p) for p in r["predict"]([[1.5, 8.5], [8.5, 1.5]])]
    [0, 1]

    Each split sees ceil(sqrt(2)) = 2 of the 2 columns here, and 2 of 4
    on a wider input:

    >>> int(r["max_features"])
    2
    >>> int(geron_random_forest([[1.0] * 4, [2.0] * 4], [0, 1], n_estimators=2)["max_features"])
    2

    Regression forests average instead of voting:

    >>> g = geron_random_forest([[1.0], [2.0], [3.0], [4.0]], [1.0, 1.0, 5.0, 5.0],
    ...                         n_estimators=7, seed=2, task="regression")
    >>> bool(g["mse"] < 1.0)
    True

    References
    ----------
    Geron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_random_forest: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n, d = A.shape
    if yv.size != n:
        raise ValueError(f"geron_random_forest: X has {n} rows but y has {yv.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("geron_random_forest: inputs contain non-finite values")
    M = int(n_estimators)
    if M < 1:
        raise ValueError(f"geron_random_forest: n_estimators must be >= 1, got {n_estimators!r}")
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"geron_random_forest: task must be auto, classification or regression, got {task!r}")
    classify = task == "classification" or (task == "auto" and np.unique(yv).size <= max(2, int(np.sqrt(n))) and np.all(yv == np.round(yv)))

    if isinstance(max_features, str):
        if max_features == "sqrt":
            k = int(np.ceil(np.sqrt(d)))
        elif max_features == "log2":
            k = max(1, int(np.ceil(np.log2(d))))
        elif max_features == "all":
            k = d
        else:
            raise ValueError(f"geron_random_forest: max_features must be sqrt, log2, all, an int or a fraction, got {max_features!r}")
    elif isinstance(max_features, float):
        k = max(1, int(round(max_features * d)))
    else:
        k = int(max_features)
    if not (1 <= k <= d):
        raise ValueError(f"geron_random_forest: max_features resolves to {k}, outside [1, {d}]")

    criterion = "gini" if classify else "mse"
    trees = []
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    votes = np.zeros((M, n))
    for m in range(M):
        rows = _lcg_bootstrap(n, seed + 7919 * m)
        col_seed = [seed + 104729 * m + 13]

        def columns_fn(depth, _s=col_seed, _d=d, _k=k):
            _s[0] = (1664525 * _s[0] + 1013904223) % 2**32
            return _lcg_cols(_d, _k, _s[0] + depth)

        t = _grow(A[rows], yv[rows], 0, int(max_depth), int(min_samples_leaf), criterion, columns_fn)
        trees.append(t)
        votes[m] = _predict_tree(t, A)
        oob = np.setdiff1d(np.arange(n), np.unique(rows))
        if oob.size:
            oob_sum[oob] += votes[m][oob]
            oob_cnt[oob] += 1

    def aggregate(P, _c=classify):
        if not _c:
            return P.mean(axis=0)
        out = np.empty(P.shape[1])
        for i in range(P.shape[1]):
            vals, cnt = np.unique(P[:, i], return_counts=True)
            out[i] = vals[np.argmax(cnt)]
        return out

    def predict(Xnew, _trees=trees, _d=d):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return aggregate(np.vstack([_predict_tree(t, B) for t in _trees]))

    pred = aggregate(votes)
    has = oob_cnt > 0
    if classify:
        score = float(np.mean(pred == yv))
        oob_pred = np.where(has, oob_sum / np.maximum(oob_cnt, 1), np.nan)
        oob_score = float(np.mean((oob_pred[has] >= 0.5).astype(float) == yv[has])) if has.any() else float("nan")
        key, label = "accuracy", "Training accuracy"
    else:
        score = float(np.mean((pred - yv) ** 2))
        oob_pred = np.where(has, oob_sum / np.maximum(oob_cnt, 1), np.nan)
        oob_score = float(np.mean((oob_pred[has] - yv[has]) ** 2)) if has.any() else float("nan")
        key, label = "mse", "Training MSE"

    imp = np.zeros(d)

    def _acc(node):
        if node["leaf"]:
            return
        imp[node["feature"]] += node["n"]
        _acc(node["left"])
        _acc(node["right"])

    for t in trees:
        _acc(t)
    if imp.sum() > 0:
        imp = imp / imp.sum()

    return RichResult(
        title="Random forest",
        summary_lines=[("Trees", M), ("Features per split", k), (label, score)],
        interpretation="Random splits make each tree worse and the ensemble better by decorrelating their errors.",
        payload={
            key: score,
            "predict": predict,
            "predictions": pred,
            "oob_score": oob_score,
            "trees": trees,
            "feature_importance": imp,
            "max_features": k,
            "task": "classification" if classify else "regression",
            "estimate": pred,
            "n": int(n),
            "method": f"Random forest ({criterion} splits) over LCG bootstraps with per-split column sampling",
        },
    )


def cheatsheet():
    return "hmrfc: Random forest, bagged trees with random split features"
