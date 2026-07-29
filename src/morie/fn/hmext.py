# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extra-trees: randomize thresholds per feature split for extra variance reduction."""

import numpy as np

from ._richresult import RichResult
from .grcart import geron_cart_split_cost
from .hmcart import geron_cart_algorithm, predict_tree

__all__ = ["geron_extra_trees"]


def _uniforms(n, state):
    out = np.empty(n)
    s = state[0]
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32
    state[0] = s
    return out


def _grow_extra(X, y, criterion, max_depth, max_features, min_samples_leaf, depth, state, stats):
    n_classes = np.unique(y).size
    pure = (n_classes < 2) if criterion != "mse" else bool(np.all(y == y[0]))
    if pure or (max_depth is not None and depth >= max_depth) or y.size < 2 * min_samples_leaf:
        stats["leaves"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        if criterion == "mse":
            return {"leaf": True, "value": float(np.mean(y)), "n": int(y.size), "depth": depth}
        vals, cnt = np.unique(y, return_counts=True)
        return {"leaf": True, "value": vals[int(np.argmax(cnt))].item(), "n": int(y.size), "depth": depth}

    n_feat = X.shape[1]
    k = min(max_features, n_feat)
    perm = np.argsort(_uniforms(n_feat, state))[:k]
    best = None
    for f in perm:
        lo, hi = float(X[:, f].min()), float(X[:, f].max())
        if hi <= lo:
            continue
        t = lo + (hi - lo) * float(_uniforms(1, state)[0])
        if t >= hi:
            t = (lo + hi) / 2.0
        left = X[:, f] <= t
        if int(left.sum()) < min_samples_leaf or int((~left).sum()) < min_samples_leaf:
            continue
        cost = float(geron_cart_split_cost(X, y, feature=int(f), threshold=t, criterion=criterion)["cost"])
        if best is None or cost < best[0]:
            best = (cost, int(f), t)
    if best is None:
        stats["leaves"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        if criterion == "mse":
            return {"leaf": True, "value": float(np.mean(y)), "n": int(y.size), "depth": depth}
        vals, cnt = np.unique(y, return_counts=True)
        return {"leaf": True, "value": vals[int(np.argmax(cnt))].item(), "n": int(y.size), "depth": depth}

    _, f, t = best
    mask = X[:, f] <= t
    stats["splits"] += 1
    return {
        "leaf": False,
        "feature": f,
        "threshold": t,
        "n": int(y.size),
        "depth": depth,
        "left": _grow_extra(X[mask], y[mask], criterion, max_depth, max_features, min_samples_leaf, depth + 1, state, stats),
        "right": _grow_extra(X[~mask], y[~mask], criterion, max_depth, max_features, min_samples_leaf, depth + 1, state, stats),
    }


def geron_extra_trees(X, y, n_estimators=10, max_features=None, seed=0, criterion="gini", max_depth=None, min_samples_leaf=1):
    """
    Extra-trees: randomize thresholds per feature split for extra variance
    reduction.

    Formula: split uses random threshold within feature range

    The difference from a random forest is exactly one line of the split
    rule, and it is implemented here: instead of scanning every candidate
    threshold, one threshold is drawn uniformly from ``[min, max]`` of
    each of ``max_features`` randomly chosen features, and the best of
    those few random candidates wins. Scoring the candidates is DELEGATED
    to :func:`morie.fn.grcart.geron_cart_split_cost`.

    That is strictly more randomisation than bagging, which is why
    extra-trees trade a little bias for a lot of variance reduction and
    are much cheaper -- no sorting of thresholds at all. Each tree sees
    the whole training set, not a bootstrap sample.

    A fully deterministic CART tree is grown alongside (via ``hmcart``) so
    the ensemble's accuracy can be compared with the greedy single tree.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    n_estimators : int, default 10
    max_features : int, optional
        Features considered per split; default ``sqrt(n)`` for
        classification, ``n`` for regression.
    seed : int, default 0
    criterion : {"gini", "entropy", "mse"}, default "gini"
    max_depth : int, optional
    min_samples_leaf : int, default 1

    Returns
    -------
    result : RichResult
        Keys: predictions, tree_predictions, trees, train_score,
        single_tree_score, disagreement, n_estimators, estimate, n,
        method.

    Examples
    --------
    A separable one-feature problem is learned by the ensemble even though
    each tree's thresholds are random:

    >>> X = [[1.0], [2.0], [8.0], [9.0]]
    >>> y = [0, 0, 1, 1]
    >>> r = geron_extra_trees(X, y, n_estimators=9, seed=5)
    >>> r["train_score"]
    1.0
    >>> len(r["tree_predictions"]), r["n_estimators"]
    (9, 9)

    Individual trees disagree -- that is the randomisation working, and
    averaging is what removes it:

    >>> r2 = geron_extra_trees([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
    ...                        [0, 1, 0, 1, 1, 0], n_estimators=11, seed=2, max_depth=1)
    >>> round(r2["disagreement"], 6)
    0.287879
    >>> round(r2["train_score"], 6)
    0.666667

    Regression averages the leaf means over trees:

    >>> r3 = geron_extra_trees([[1.0], [2.0], [8.0], [9.0]], [0.0, 0.0, 4.0, 4.0],
    ...                        n_estimators=7, seed=1, criterion="mse")
    >>> r3["train_mse"] < 1.0
    True

    References
    ----------
    Géron Ch 6
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    ya = np.asarray(y).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_extra_trees: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    if Xa.size == 0:
        raise ValueError("geron_extra_trees: X is empty")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_extra_trees: X contains non-finite values")
    if criterion not in ("gini", "entropy", "mse"):
        raise ValueError(f"geron_extra_trees: criterion must be 'gini', 'entropy' or 'mse', got {criterion!r}")
    if criterion == "mse":
        ya = ya.astype(float)
    B = int(n_estimators)
    if B < 1:
        raise ValueError(f"geron_extra_trees: n_estimators must be >= 1, got {n_estimators!r}")
    n_feat = Xa.shape[1]
    if max_features is None:
        mf = n_feat if criterion == "mse" else max(1, int(np.sqrt(n_feat)))
    else:
        mf = int(max_features)
        if not (1 <= mf <= n_feat):
            raise ValueError(f"geron_extra_trees: max_features must lie in 1..{n_feat}, got {max_features!r}")
    msl = int(min_samples_leaf)
    if msl < 1:
        raise ValueError(f"geron_extra_trees: min_samples_leaf must be >= 1, got {min_samples_leaf!r}")

    state = [int(seed) % 2**32]
    trees, tree_preds = [], []
    stats = {"leaves": 0, "splits": 0, "max_depth": 0}
    for _ in range(B):
        t = _grow_extra(Xa, ya, criterion, None if max_depth is None else int(max_depth), mf, msl, 0, state, stats)
        trees.append(t)
        tree_preds.append(predict_tree(t, Xa))

    P = np.asarray(tree_preds)
    single = geron_cart_algorithm(Xa, ya, criterion=criterion, max_depth=max_depth)

    payload = {
        "trees": trees,
        "tree_predictions": [list(p) for p in tree_preds],
        "n_estimators": B,
        "max_features": mf,
        "criterion": criterion,
        "n_leaves_total": int(stats["leaves"]),
        "n": int(Xa.shape[0]),
        "method": "extra-trees with uniformly random per-feature thresholds; split cost delegated to grcart",
    }
    if criterion == "mse":
        pred = P.astype(float).mean(axis=0)
        mse = float(np.mean((pred - ya) ** 2))
        payload.update(
            predictions=pred.tolist(),
            train_mse=mse,
            train_score=mse,
            single_tree_score=float(single["train_mse"]),
            disagreement=float(np.mean(P.astype(float).var(axis=0))),
            estimate=mse,
        )
        head = ("Train MSE", mse)
    else:
        classes = np.unique(ya)
        counts = np.stack([(P == c).sum(axis=0) for c in classes], axis=1)
        pred = classes[counts.argmax(axis=1)]
        acc = float(np.mean(pred == ya))
        payload.update(
            predictions=pred.tolist(),
            train_score=acc,
            train_accuracy=acc,
            single_tree_score=float(single["train_accuracy"]),
            disagreement=float(np.mean(1.0 - counts.max(axis=1) / B)),
            estimate=acc,
        )
        head = ("Train accuracy", acc)

    return RichResult(
        title="Extra-trees ensemble",
        summary_lines=[head, ("Trees", B), ("Features per split", mf)],
        interpretation="Random thresholds add bias to each tree but cut ensemble variance, and cost no sorting at all.",
        payload=payload,
    )


def cheatsheet():
    return "hmext: Extra-trees: randomize thresholds per feature split for extra variance reduction"
