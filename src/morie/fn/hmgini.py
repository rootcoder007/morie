# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gini impurity for a node."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gini_impurity"]


def geron_gini_impurity(y):
    """
    Gini impurity for a node.

    Formula: G = 1 - sum_k p_k^2

    The class proportions are counted exactly, so a pure node scores 0 and
    a node with ``K`` equally frequent classes scores ``1 - 1/K`` -- which
    is the maximum attainable for that many classes and is returned as
    ``max_possible`` for calibration.

    Parameters
    ----------
    y : array-like
        Class labels at the node (any hashable/comparable dtype).

    Returns
    -------
    result : RichResult
        Keys: gini, proportions, classes, counts, max_possible,
        estimate, n, method.

    Examples
    --------
    >>> round(geron_gini_impurity([0, 0, 1, 1])["gini"], 12)
    0.5
    >>> geron_gini_impurity([1, 1, 1])["gini"]
    0.0
    >>> round(geron_gini_impurity([0, 0, 0, 1])["gini"], 12)
    0.375
    >>> r = geron_gini_impurity(["a", "b", "c"])
    >>> round(r["gini"], 12), round(r["max_possible"], 12)
    (0.666666666667, 0.666666666667)

    References
    ----------
    Géron Ch 5
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("geron_gini_impurity: y is empty; impurity is undefined for an empty node")
    classes, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    gini = float(1.0 - np.sum(p * p))
    K = int(classes.size)
    return RichResult(
        title="Gini impurity",
        summary_lines=[("Gini", gini), ("Classes", K)],
        interpretation="0 means the node is pure; the ceiling for K classes is 1 - 1/K.",
        payload={
            "gini": gini,
            "proportions": p.tolist(),
            "classes": classes.tolist(),
            "counts": counts.astype(int).tolist(),
            "n_classes": K,
            "max_possible": float(1.0 - 1.0 / K),
            "estimate": gini,
            "n": int(y.size),
            "method": "Gini impurity G = 1 - sum_k p_k^2",
        },
    )


def cheatsheet():
    return "hmgini: Gini impurity for a node"
