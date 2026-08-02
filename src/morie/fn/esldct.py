# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CART regression tree (ESL Ch 9.2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_decision_tree", "esl_tree_predict"]


def _best_split(X, y, min_leaf):
    """Exhaustive axis-aligned split minimising within-node SSE."""
    n, p = X.shape
    best = None
    parent = float(np.sum((y - y.mean()) ** 2))
    for j in range(p):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys = X[order, j], y[order]
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue                       # cannot split inside a tie
            L, R = ys[:i], ys[i:]
            sse = float(np.sum((L - L.mean()) ** 2) + np.sum((R - R.mean()) ** 2))
            if best is None or sse < best[0] - 1e-15:
                best = (sse, j, 0.5 * (xs[i - 1] + xs[i]), parent - sse)
    return best


def esl_decision_tree(X, y, max_depth=3, min_leaf=1, min_impurity_decrease=0.0):
    """
    CART regression tree grown by greedy binary splitting.

    Formula: at each node choose (j, s) minimising
    sum_{x in L} (y - c_L)^2 + sum_{x in R} (y - c_R)^2, with c_L and
    c_R the region means, which are the least squares constants.
    Splits are midpoints between consecutive distinct values, and a
    tie in the feature cannot be split through — otherwise identical
    rows would land in different leaves.

    Growth stops on max_depth, min_leaf, a pure node, or an impurity
    decrease below the threshold. ESL Ch 9.2 recommends growing large
    then cost-complexity pruning; this grows with stopping rules
    instead, which is stated because the two give different trees and
    the difference matters when someone compares against rpart.

    The returned tree is a plain nested dict, usable directly:
      leaf   {"leaf": True, "value": float, "n": int}
      split  {"leaf": False, "feature": int, "threshold": float,
              "left": <node>, "right": <node>, "n": int,
              "impurity_decrease": float}
    Feed it to esl_tree_predict to score new data.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Numeric response.
    max_depth : int
        Maximum depth, >= 1.
    min_leaf : int
        Minimum observations in a leaf, >= 1.
    min_impurity_decrease : float
        Minimum SSE reduction required to accept a split.

    Returns
    -------
    result : dict
        Keys: estimate (training RSS), tree, n_leaves, depth,
        fitted, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 9.2.2 (Eq. 9.13-9.15).

    Examples
    --------
    A clean step function is recovered by one split:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [1.0, 1.0, 5.0, 5.0]
    >>> out = esl_decision_tree(X, y, max_depth=1)
    >>> out["tree"]["feature"], out["tree"]["threshold"]
    (0, 1.5)
    >>> out["tree"]["left"]["value"], out["tree"]["right"]["value"]
    (1.0, 5.0)
    >>> out["estimate"]
    0.0
    >>> esl_tree_predict(out["tree"], [[0.5], [2.5]])
    [1.0, 5.0]

    A constant response yields a single leaf:

    >>> esl_decision_tree(X, [2.0, 2.0, 2.0, 2.0])["n_leaves"]
    1
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    max_depth = int(max_depth)
    min_leaf = int(min_leaf)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}.")
    if min_leaf < 1:
        raise ValueError(f"min_leaf must be >= 1; got {min_leaf}.")

    def grow(idx, depth):
        yy = y[idx]
        node_n = int(idx.size)
        if (depth >= max_depth or node_n < 2 * min_leaf
                or float(np.ptp(yy)) == 0.0):
            return {"leaf": True, "value": float(yy.mean()), "n": node_n}
        best = _best_split(X[idx], yy, min_leaf)
        if best is None or best[3] <= float(min_impurity_decrease):
            return {"leaf": True, "value": float(yy.mean()), "n": node_n}
        _, j, thr, gain = best
        mask = X[idx, j] <= thr
        return {"leaf": False, "feature": int(j), "threshold": float(thr),
                "n": node_n, "impurity_decrease": float(gain),
                "left": grow(idx[mask], depth + 1),
                "right": grow(idx[~mask], depth + 1)}

    tree = grow(np.arange(n), 0)
    fitted = np.asarray(esl_tree_predict(tree, X), dtype=float)
    resid = y - fitted

    def count(node):
        return 1 if node["leaf"] else count(node["left"]) + count(node["right"])

    def deep(node):
        return 0 if node["leaf"] else 1 + max(deep(node["left"]), deep(node["right"]))

    return RichResult(payload={
        "estimate": float(resid @ resid), "tree": tree,
        "n_leaves": count(tree), "depth": deep(tree),
        "fitted": [float(v) for v in fitted], "n": int(n), "p": int(p),
        "method": "CART regression tree, greedy SSE splits, stopping rules (not pruned)"})


def esl_tree_predict(tree, X):
    """
    Score new rows through a tree returned by [esl_decision_tree].

    Parameters
    ----------
    tree : dict
        Node dict as documented on esl_decision_tree.
    X : array-like, shape (m, p)
        Rows to predict.

    Returns
    -------
    list of float
        One prediction per row.

    Examples
    --------
    >>> t = esl_decision_tree([[0.0], [1.0], [9.0]], [0.0, 0.0, 4.0], max_depth=1)["tree"]
    >>> esl_tree_predict(t, [[0.0], [100.0]])
    [0.0, 4.0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = []
    for row in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        out.append(float(node["value"]))
    return out


def cheatsheet():
    return "esldct: greedy SSE tree + esl_tree_predict; stopping rules, not cost-complexity pruning"
