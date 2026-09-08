# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression decision tree (CART) minimising the MSE per leaf."""

from . import _array_core as np

from ._richresult import RichResult
from .grcart import geron_cart_split_cost

__all__ = ["geron_regression_tree"]


def _leaf_value(y, criterion):
    if criterion == "mse":
        return float(np.mean(y))
    vals, cnt = np.unique(y, return_counts=True)
    return float(vals[np.argmax(cnt)])


def _best_split(X, y, criterion, columns, min_samples_leaf):
    """Greedy CART split; the per-candidate cost is delegated to grcart."""
    best = None
    for j in columns:
        col = X[:, j]
        vals = np.unique(col)
        if vals.size < 2:
            continue
        for thr in (vals[:-1] + vals[1:]) / 2.0:
            left = col <= thr
            if left.sum() < min_samples_leaf or (~left).sum() < min_samples_leaf:
                continue
            cost = float(geron_cart_split_cost(X, y, j, thr, criterion=criterion)["cost"])
            if best is None or cost < best[0]:
                best = (cost, int(j), float(thr))
    return best


def _grow(X, y, depth, max_depth, min_samples_leaf, criterion, columns_fn):
    """Recursively grow a CART node. ``columns_fn(depth)`` supplies candidate columns."""
    if depth >= max_depth or y.size < 2 * min_samples_leaf or np.unique(y).size == 1:
        return {"leaf": True, "value": _leaf_value(y, criterion), "n": int(y.size)}
    best = _best_split(X, y, criterion, columns_fn(depth), min_samples_leaf)
    if best is None:
        return {"leaf": True, "value": _leaf_value(y, criterion), "n": int(y.size)}
    _, j, thr = best
    left = X[:, j] <= thr
    return {
        "leaf": False,
        "feature": j,
        "threshold": thr,
        "n": int(y.size),
        "left": _grow(X[left], y[left], depth + 1, max_depth, min_samples_leaf, criterion, columns_fn),
        "right": _grow(X[~left], y[~left], depth + 1, max_depth, min_samples_leaf, criterion, columns_fn),
    }


def _predict_tree(node, X):
    out = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        nd = node
        while not nd["leaf"]:
            nd = nd["left"] if X[i, nd["feature"]] <= nd["threshold"] else nd["right"]
        out[i] = nd["value"]
    return out


def _leaves(node, acc=None):
    acc = [] if acc is None else acc
    if node["leaf"]:
        acc.append(node)
    else:
        _leaves(node["left"], acc)
        _leaves(node["right"], acc)
    return acc


def geron_regression_tree(X, y, max_depth=3, min_samples_leaf=1):
    """
    Regression decision tree via CART minimising the MSE per leaf.

    Formula: split minimising (m_L/m)*MSE_L + (m_R/m)*MSE_R

    Each split cost is DELEGATED to the finished
    :func:`morie.fn.grcart.geron_cart_split_cost`, so the criterion here
    is exactly the book's. The prediction of a leaf is the MEAN of its
    training targets, which makes the tree a piecewise-constant fit: it
    cannot extrapolate a trend, and outside the training range it repeats
    the nearest leaf's mean forever.

    Left unregularised the tree fits every point and generalises badly,
    which is why ``max_depth`` and ``min_samples_leaf`` are the first two
    arguments a user should reach for.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    max_depth : int, default 3
    min_samples_leaf : int, default 1

    Returns
    -------
    result : RichResult
        Keys: tree, predict, predictions, mse, n_leaves, depth,
        feature_importance, estimate, n, method.

    Examples
    --------
    A step function is fit exactly by one split at 2.5:

    >>> r = geron_regression_tree([[1.0], [2.0], [3.0], [4.0]], [1.0, 1.0, 5.0, 5.0], max_depth=1)
    >>> float(r["tree"]["threshold"]), int(r["n_leaves"])
    (2.5, 2)
    >>> [float(v) for v in r["predictions"]]
    [1.0, 1.0, 5.0, 5.0]
    >>> float(r["mse"])
    0.0

    Being piecewise constant, it predicts the nearest leaf's mean far
    outside the training range:

    >>> float(r["predict"]([[100.0]])[0])
    5.0

    References
    ----------
    Geron Ch 5
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_regression_tree: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_regression_tree: X has {A.shape[0]} rows but y has {yv.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)):
        raise ValueError("geron_regression_tree: inputs contain non-finite values")
    D = int(max_depth)
    if D < 1:
        raise ValueError(f"geron_regression_tree: max_depth must be >= 1, got {max_depth!r}")
    L = int(min_samples_leaf)
    if L < 1:
        raise ValueError(f"geron_regression_tree: min_samples_leaf must be >= 1, got {min_samples_leaf!r}")

    cols = np.arange(A.shape[1])
    tree = _grow(A, yv, 0, D, L, "mse", lambda _d: cols)
    pred = _predict_tree(tree, A)
    mse = float(np.mean((pred - yv) ** 2))

    imp = np.zeros(A.shape[1])

    def _acc(node, Xs, ys):
        if node["leaf"]:
            return
        m = ys.size
        lm = Xs[:, node["feature"]] <= node["threshold"]
        gain = float(np.var(ys)) - (lm.sum() / m) * float(np.var(ys[lm])) - ((~lm).sum() / m) * float(np.var(ys[~lm]))
        imp[node["feature"]] += gain * m
        _acc(node["left"], Xs[lm], ys[lm])
        _acc(node["right"], Xs[~lm], ys[~lm])

    _acc(tree, A, yv)
    if imp.sum() > 0:
        imp = imp / imp.sum()

    def _depth(node):
        return 0 if node["leaf"] else 1 + max(_depth(node["left"]), _depth(node["right"]))

    def predict(Xnew, _t=tree, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return _predict_tree(_t, B)

    return RichResult(
        title="Regression tree (CART)",
        summary_lines=[("Depth", _depth(tree)), ("Leaves", len(_leaves(tree))), ("Training MSE", mse)],
        interpretation="Leaf means make the fit piecewise constant; it cannot extrapolate a trend.",
        payload={
            "tree": tree,
            "predict": predict,
            "predictions": pred,
            "mse": mse,
            "n_leaves": len(_leaves(tree)),
            "depth": _depth(tree),
            "feature_importance": imp,
            "estimate": pred,
            "n": int(A.shape[0]),
            "method": "CART regression tree, split cost delegated to morie.fn.grcart",
        },
    )


def cheatsheet():
    return "hmrdt: CART regression tree minimising per-leaf MSE"
