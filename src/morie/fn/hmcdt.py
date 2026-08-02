# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification decision tree via CART with Gini or entropy."""

from . import _array_core as np

from ._richresult import RichResult
from .hmcart import geron_cart_algorithm

__all__ = ["geron_classification_tree"]


def _proba_of(tree, row, classes):
    node = tree
    while not node["leaf"]:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    p = node.get("proba", {})
    return [float(p.get(c, 0.0)) for c in classes]


def geron_classification_tree(X, y, criterion="gini", max_depth=None, min_samples_leaf=1):
    """
    Classification decision tree via CART with Gini or entropy.

    Formula: recursively split to minimize impurity until stopping criteria

    Growing is DELEGATED to
    :func:`morie.fn.hmcart.geron_cart_algorithm` (which in turn delegates
    the split cost to ``grcart``); this module restricts the criterion to
    the two classification impurities and adds what a classifier owes its
    caller: class probabilities from the leaf class frequencies.

    Gini and entropy usually pick the same split. Where they differ,
    entropy is marginally more willing to isolate a rare class, since
    ``-p log p`` falls off more slowly near 0 than ``1 - p^2``; both are
    available so the difference can be measured rather than argued about.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Class labels.
    criterion : {"gini", "entropy"}, default "gini"
    max_depth : int, optional
    min_samples_leaf : int, default 1

    Returns
    -------
    result : RichResult
        Keys: tree, predictions, probabilities, classes, n_leaves, depth,
        train_accuracy, feature_importances, estimate, n, method.

    Examples
    --------
    >>> r = geron_classification_tree([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1])
    >>> r["predictions"]
    [0, 0, 1, 1]
    >>> r["classes"]
    [0, 1]
    >>> r["probabilities"][0]
    [1.0, 0.0]
    >>> r["train_accuracy"], r["n_leaves"]
    (1.0, 2)

    A depth-limited tree on a mixed leaf returns the leaf frequencies, so
    the probabilities are not degenerate:

    >>> r2 = geron_classification_tree([[1.0], [2.0], [3.0]], [0, 1, 1], max_depth=0)
    >>> [round(v, 6) for v in r2["probabilities"][0]]
    [0.333333, 0.666667]
    >>> r2["n_leaves"]
    1

    References
    ----------
    Géron Ch 5
    """
    if criterion not in ("gini", "entropy"):
        raise ValueError(
            f"geron_classification_tree: criterion must be 'gini' or 'entropy', got {criterion!r} "
            "(use hmcart with criterion='mse' for regression)"
        )
    ya = np.asarray(y).ravel()
    if ya.size and np.issubdtype(ya.dtype, np.floating) and not np.all(ya == np.round(ya)):
        raise ValueError("geron_classification_tree: y looks continuous; a classification tree needs discrete labels")

    base = geron_cart_algorithm(
        X, y, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    classes = [c.item() for c in np.unique(ya)]
    proba = [_proba_of(base["tree"], row, classes) for row in Xa]

    return RichResult(
        title="Classification tree",
        summary_lines=[("Train accuracy", float(base["train_accuracy"])), ("Leaves", int(base["n_leaves"])), ("Criterion", criterion)],
        interpretation="Leaf class frequencies are the predicted probabilities; a pure leaf therefore predicts with probability 1.",
        payload={
            "tree": base["tree"],
            "predictions": base["predictions"],
            "probabilities": proba,
            "classes": classes,
            "n_leaves": int(base["n_leaves"]),
            "depth": int(base["depth"]),
            "n_splits": int(base["n_splits"]),
            "train_accuracy": float(base["train_accuracy"]),
            "feature_importances": base["feature_importances"],
            "criterion": criterion,
            "estimate": float(base["train_accuracy"]),
            "n": int(base["n"]),
            "method": "CART classification tree (growth delegated to hmcart) with leaf-frequency probabilities",
        },
    )


def cheatsheet():
    return "hmcdt: Classification decision tree via CART with Gini or entropy"
