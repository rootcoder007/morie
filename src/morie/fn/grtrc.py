# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification tree leaf prediction (majority class)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_tree_classification_leaf"]

_METHOD = "Classification tree leaf (majority class)"


def geron_tree_classification_leaf(y, leaf_mask=None):
    r"""Majority class of the instances that reach a leaf.

    .. math::
        \hat y_{\text{leaf}} = \arg\max_k \sum_{i \in \text{leaf}}
            \mathbb{1}\{y_i = k\}

    The class *proportions* in the leaf are what a decision tree reports
    as its predicted probabilities -- so a leaf holding 5 of one class
    and 5 of another predicts 50/50, and one holding a single instance
    predicts 100%, which is precisely how unpruned trees end up wildly
    overconfident.  Gini impurity is returned alongside because it is the
    quantity the split search was minimising to produce this leaf.

    Parameters
    ----------
    y : array-like of int
        Class labels of the training set.
    leaf_mask : array-like of bool, optional
        Which instances reach the leaf. Defaults to all of them.

    Returns
    -------
    RichResult
        Payload keys ``prediction``, ``proportions``, ``counts``,
        ``gini``, ``entropy``, ``n_leaf``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 5, Classification Trees section.

    Examples
    --------
    Géron's iris leaf: 0 setosa, 49 versicolor, 5 virginica.

    >>> y = [1] * 49 + [2] * 5
    >>> r = geron_tree_classification_leaf(y)
    >>> r["prediction"]
    1
    >>> round(r["gini"], 6)
    0.168038

    A masked leaf sees only its own instances:

    >>> m = geron_tree_classification_leaf([0, 0, 1, 1, 1], [True, True, False, False, False])
    >>> m["prediction"], m["gini"]
    (0, 0.0)
    """
    yv = np.asarray(y).ravel()
    if yv.size == 0:
        raise ValueError("y is empty.")
    if not np.all(yv == np.round(np.asarray(yv, dtype=float))):
        raise ValueError("classification leaves need integer class labels.")
    yv = yv.astype(int)
    if yv.min() < 0:
        raise ValueError(f"class labels must be non-negative, got {int(yv.min())}.")
    K = int(yv.max()) + 1

    if leaf_mask is None:
        sel = yv
    else:
        mask = np.asarray(leaf_mask)
        if mask.shape != yv.shape:
            raise ValueError(f"leaf_mask has shape {mask.shape} but y has {yv.shape}.")
        sel = yv[mask.astype(bool)]
        if sel.size == 0:
            raise ValueError("leaf_mask selects no instances; an empty leaf has no prediction.")

    counts = np.bincount(sel, minlength=K)
    p = counts / sel.size
    pred = int(np.argmax(counts))
    gini = float(1.0 - np.sum(p**2))
    nz = p[p > 0]
    ent = float(-np.sum(nz * np.log2(nz)))

    return RichResult(
        title="Classification tree leaf",
        summary_lines=[("Prediction", pred), ("Leaf size", int(sel.size)), ("Gini", gini)],
        payload={
            "prediction": pred,
            "proportions": p.tolist(),
            "counts": counts.astype(int).tolist(),
            "gini": gini,
            "entropy": ent,
            "n_leaf": int(sel.size),
            "estimate": pred,
            "n": int(yv.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtrc: leaf predicts the majority class; proportions are the tree's probabilities; Gini reported"
