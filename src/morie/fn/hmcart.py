# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CART algorithm: greedy binary splits minimizing impurity."""

from . import _array_core as np

from ._richresult import RichResult
from .grcart import geron_cart_split_cost

__all__ = ["geron_cart_algorithm", "predict_tree"]


def _leaf(y, criterion):
    y = np.asarray(y)
    if criterion == "mse":
        yv = y.astype(float)
        return {"leaf": True, "value": float(yv.mean()), "n": int(y.size), "impurity": float(np.mean((yv - yv.mean()) ** 2))}
    classes, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    imp = float(1.0 - np.sum(p * p)) if criterion == "gini" else float(-np.sum(p * np.log2(p)))
    return {
        "leaf": True,
        "value": classes[int(np.argmax(counts))].item(),
        "proba": dict(zip([c.item() for c in classes], p.tolist())),
        "n": int(y.size),
        "impurity": imp,
    }


def _best_split(X, y, criterion, min_samples_leaf):
    best = None
    m, n = X.shape
    for k in range(n):
        col = X[:, k]
        vals = np.unique(col)
        if vals.size < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            left = col <= t
            nl, nr = int(left.sum()), int(m - left.sum())
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            res = geron_cart_split_cost(X, y, feature=k, threshold=float(t), criterion=criterion)
            cost = float(res["cost"])
            if best is None or cost < best["cost"] - 1e-15:
                best = {
                    "cost": cost,
                    "feature": k,
                    "threshold": float(t),
                    "impurity_decrease": float(res["impurity_decrease"]),
                    "n_left": nl,
                    "n_right": nr,
                }
    return best


def _grow(X, y, criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, depth, stats):
    node_imp = _leaf(y, criterion)["impurity"]
    stop = (
        (max_depth is not None and depth >= max_depth)
        or y.size < min_samples_split
        or node_imp <= 0.0
    )
    if not stop:
        best = _best_split(X, y, criterion, min_samples_leaf)
        if best is not None and not (best["impurity_decrease"] < min_impurity_decrease):
            mask = X[:, best["feature"]] <= best["threshold"]
            stats["splits"] += 1
            node = {
                "leaf": False,
                "feature": best["feature"],
                "threshold": best["threshold"],
                "impurity": node_imp,
                "impurity_decrease": best["impurity_decrease"],
                "n": int(y.size),
                "depth": depth,
                "left": _grow(X[mask], y[mask], criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, depth + 1, stats),
                "right": _grow(X[~mask], y[~mask], criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, depth + 1, stats),
            }
            return node
    lf = _leaf(y, criterion)
    lf["depth"] = depth
    stats["leaves"] += 1
    stats["max_depth"] = max(stats["max_depth"], depth)
    return lf


def predict_tree(tree, X):
    """Route each row of ``X`` down ``tree`` and return the leaf values."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = []
    for row in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        out.append(node["value"])
    return out


def geron_cart_algorithm(
    X,
    y,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
):
    """
    CART algorithm: greedy binary splits minimizing impurity.

    Formula: split minimizing J(k,t_k) = (m_L/m)*G_L + (m_R/m)*G_R

    The tree is grown for real. At every node each feature's midpoint
    thresholds are enumerated and scored, with the cost function itself
    DELEGATED to :func:`morie.fn.grcart.geron_cart_split_cost`, so this
    module never re-derives the impurity arithmetic.

    Greedy means no lookahead: the split that looks best now is taken and
    never revisited, which is why CART cannot represent an XOR with one
    split even though two splits capture it exactly. Growth stops on any
    of: pure node, ``max_depth``, ``min_samples_split``, or no candidate
    split that leaves ``min_samples_leaf`` on both sides and improves
    impurity by more than ``min_impurity_decrease``.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Labels, or targets with ``criterion="mse"``.
    criterion : {"gini", "entropy", "mse"}, default "gini"
    max_depth : int, optional
    min_samples_split : int, default 2
    min_samples_leaf : int, default 1
    min_impurity_decrease : float, default 0.0

    Returns
    -------
    result : RichResult
        Keys: tree, predictions, n_leaves, depth, n_splits,
        train_accuracy or train_mse, feature_importances, estimate,
        n, method.

    Examples
    --------
    A one-feature problem separable at 2.5 needs a single split:

    >>> r = geron_cart_algorithm([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1])
    >>> r["tree"]["feature"], r["tree"]["threshold"]
    (0, 2.5)
    >>> r["n_leaves"], r["depth"], r["train_accuracy"]
    (2, 1, 1.0)
    >>> r["predictions"]
    [0, 0, 1, 1]

    XOR defeats a depth-1 tree but not a depth-2 one -- the greedy
    limitation, measured rather than asserted:

    >>> Xx = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> yx = [0, 1, 1, 0]
    >>> geron_cart_algorithm(Xx, yx, max_depth=1)["train_accuracy"]
    0.5
    >>> geron_cart_algorithm(Xx, yx, max_depth=2)["train_accuracy"]
    1.0

    Regression trees fit the mean of each leaf:

    >>> r2 = geron_cart_algorithm([[1.0], [2.0], [3.0], [4.0]], [1.0, 1.0, 5.0, 5.0],
    ...                           criterion="mse", max_depth=1)
    >>> r2["predictions"]
    [1.0, 1.0, 5.0, 5.0]
    >>> r2["train_mse"]
    0.0

    References
    ----------
    Géron Ch 5
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError(f"geron_cart_algorithm: X must be a non-empty 2-D array, got shape {Xa.shape}")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_cart_algorithm: X contains non-finite values")
    ya = np.asarray(y).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_cart_algorithm: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    if criterion not in ("gini", "entropy", "mse"):
        raise ValueError(f"geron_cart_algorithm: criterion must be 'gini', 'entropy' or 'mse', got {criterion!r}")
    if criterion == "mse":
        ya = ya.astype(float)
        if not np.all(np.isfinite(ya)):
            raise ValueError("geron_cart_algorithm: y contains non-finite values")
    md = None if max_depth is None else int(max_depth)
    if md is not None and md < 0:
        raise ValueError(f"geron_cart_algorithm: max_depth must be non-negative, got {max_depth!r}")
    mss, msl = int(min_samples_split), int(min_samples_leaf)
    if mss < 2:
        raise ValueError(f"geron_cart_algorithm: min_samples_split must be >= 2, got {min_samples_split!r}")
    if msl < 1:
        raise ValueError(f"geron_cart_algorithm: min_samples_leaf must be >= 1, got {min_samples_leaf!r}")
    mid = float(min_impurity_decrease)
    if mid < 0:
        raise ValueError(f"geron_cart_algorithm: min_impurity_decrease must be non-negative, got {min_impurity_decrease!r}")

    stats = {"leaves": 0, "splits": 0, "max_depth": 0}
    tree = _grow(Xa, ya, criterion, md, mss, msl, mid, 0, stats)
    preds = predict_tree(tree, Xa)

    imp = np.zeros(Xa.shape[1])

    def walk(node):
        if node["leaf"]:
            return
        imp[node["feature"]] += node["n"] * node["impurity_decrease"]
        walk(node["left"])
        walk(node["right"])

    walk(tree)
    total = imp.sum()
    importances = (imp / total) if total > 0 else imp

    payload = {
        "tree": tree,
        "predictions": preds,
        "n_leaves": int(stats["leaves"]),
        "depth": int(stats["max_depth"]),
        "n_splits": int(stats["splits"]),
        "criterion": criterion,
        "feature_importances": importances.tolist(),
        "n": int(Xa.shape[0]),
        "method": "greedy CART; split cost delegated to grcart",
    }
    if criterion == "mse":
        mse = float(np.mean((np.asarray(preds, dtype=float) - ya) ** 2))
        payload["train_mse"] = mse
        payload["estimate"] = mse
        headline = ("Train MSE", mse)
    else:
        acc = float(np.mean(np.asarray(preds) == ya))
        payload["train_accuracy"] = acc
        payload["estimate"] = acc
        headline = ("Train accuracy", acc)

    return RichResult(
        title="CART tree",
        summary_lines=[headline, ("Leaves", stats["leaves"]), ("Depth", stats["max_depth"])],
        interpretation="CART is greedy and local: no split is ever revisited, so XOR-like structure needs extra depth.",
        payload=payload,
    )


def cheatsheet():
    return "hmcart: CART algorithm: greedy binary splits minimizing impurity"


# compact alias per ledger/NAMING.md
predicttree = predict_tree
