# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tree regularization via max_depth, min_samples_split, min_samples_leaf."""

from . import _array_core as np

from ._richresult import RichResult
from .hmcart import geron_cart_algorithm

__all__ = ["geron_tree_regularization"]


def geron_tree_regularization(X, y, max_depth=None, min_samples_leaf=1, min_samples_split=2, criterion="gini"):
    """
    Tree regularization via max_depth, min_samples_split, min_samples_leaf.

    Formula: constraints on tree structure to reduce overfitting

    Two trees are actually grown -- the constrained one and the
    unconstrained baseline -- so the effect of the hyperparameters is
    measured rather than described. Growth is DELEGATED to
    :func:`morie.fn.hmcart.geron_cart_algorithm`.

    An unconstrained CART tree keeps splitting until every leaf is pure,
    which drives training error to zero and is exactly the overfitting the
    constraints exist to prevent. ``leaves_saved`` and ``train_score_cost``
    quantify the trade: how much simpler the tree got, and how much
    training accuracy (or MSE) that cost.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    max_depth : int, optional
    min_samples_leaf : int, default 1
    min_samples_split : int, default 2
    criterion : {"gini", "entropy", "mse"}, default "gini"

    Returns
    -------
    result : RichResult
        Keys: tree, baseline_leaves, n_leaves, depth, baseline_depth,
        leaves_saved, train_score, baseline_train_score,
        train_score_cost, constraints, estimate, n, method.

    Examples
    --------
    Four separable points: unconstrained CART builds two leaves, and
    depth 0 collapses to one:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> y = [0, 0, 1, 1]
    >>> r = geron_tree_regularization(X, y, max_depth=0)
    >>> r["n_leaves"], r["baseline_leaves"]
    (1, 2)
    >>> r["train_score"], r["baseline_train_score"]
    (0.5, 1.0)
    >>> round(r["train_score_cost"], 6)
    0.5

    A leaf-size floor also prunes: with three points per leaf required, no
    split of a 4-point sample is legal:

    >>> r2 = geron_tree_regularization(X, y, min_samples_leaf=3)
    >>> r2["n_leaves"], r2["leaves_saved"]
    (1, 1)

    Nothing is lost when the constraint does not bind:

    >>> r3 = geron_tree_regularization(X, y, max_depth=5)
    >>> r3["train_score_cost"]
    0.0

    References
    ----------
    Géron Ch 5
    """
    constrained = geron_cart_algorithm(
        X,
        y,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    baseline = geron_cart_algorithm(X, y, criterion=criterion)

    key = "train_mse" if criterion == "mse" else "train_accuracy"
    score = float(constrained[key])
    base_score = float(baseline[key])
    cost = (base_score - score) if criterion != "mse" else (score - base_score)

    return RichResult(
        title="Tree regularization",
        summary_lines=[
            ("Leaves", int(constrained["n_leaves"])),
            ("Baseline leaves", int(baseline["n_leaves"])),
            (f"Train {'MSE' if criterion == 'mse' else 'accuracy'}", score),
        ],
        interpretation=(
            "An unconstrained CART tree grows until every leaf is pure; the constraints trade training "
            "fit for a smaller hypothesis space."
        ),
        payload={
            "tree": constrained["tree"],
            "predictions": constrained["predictions"],
            "n_leaves": int(constrained["n_leaves"]),
            "depth": int(constrained["depth"]),
            "baseline_leaves": int(baseline["n_leaves"]),
            "baseline_depth": int(baseline["depth"]),
            "leaves_saved": int(baseline["n_leaves"]) - int(constrained["n_leaves"]),
            "train_score": score,
            "baseline_train_score": base_score,
            "train_score_cost": float(cost),
            "constraints": {
                "max_depth": max_depth,
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
            },
            "criterion": criterion,
            "estimate": score,
            "n": int(constrained["n"]),
            "method": "constrained vs unconstrained CART, both grown via hmcart",
        },
    )


def cheatsheet():
    return "hmdtr: Tree regularization via max_depth, min_samples_split, min_samples_leaf"
