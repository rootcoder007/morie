# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decision trees are insensitive to feature scale (axis-aligned splits)."""

import numpy as np

from ._richresult import RichResult
from .hmcart import geron_cart_algorithm, predict_tree

__all__ = ["geron_tree_sensitivity_scale"]


def geron_tree_sensitivity_scale(X, y, a=100.0, b=-7.0, feature=None, criterion="gini", max_depth=None):
    """
    Decision trees are insensitive to feature scale (axis-aligned splits).

    Formula: invariant to x' = a*x + b

    The invariance is demonstrated by doing the experiment: a tree is
    grown on ``X`` and another on the affinely rescaled ``a*X + b``
    (optionally on one column only), and the two are compared. Splits are
    axis-aligned comparisons ``x_k <= t``, and a monotone map takes each
    threshold to ``a*t + b`` while preserving every ordering, so the
    predictions must be identical and the thresholds must be exactly the
    transformed ones.

    A distance-based baseline is run alongside as the control: 1-nearest
    neighbour on the same data, which *does* change its answers under
    rescaling. Without that contrast, "invariant" is unfalsifiable.

    ``a`` must be positive; a negative scale mirrors the axis, which
    swaps the children of every split -- still equivalent, but no longer
    threshold-by-threshold comparable.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    a : float, default 100.0
        Positive scale factor.
    b : float, default -7.0
        Shift.
    feature : int, optional
        Rescale only this column; default all.
    criterion : {"gini", "entropy", "mse"}, default "gini"
    max_depth : int, optional

    Returns
    -------
    result : RichResult
        Keys: predictions_match, thresholds, scaled_thresholds,
        expected_thresholds, thresholds_match, knn_predictions,
        knn_scaled_predictions, knn_match, estimate, n, method.

    Examples
    --------
    Rescaling by 100 and shifting by -7 changes nothing about the tree:

    >>> X = [[1.0, 5.0], [2.0, 4.0], [3.0, 9.0], [4.0, 1.0]]
    >>> y = [0, 0, 1, 1]
    >>> r = geron_tree_sensitivity_scale(X, y)
    >>> r["predictions_match"], r["thresholds_match"]
    (True, True)
    >>> r["thresholds"], r["scaled_thresholds"]
    ([2.5], [243.0])
    >>> r["expected_thresholds"]
    [243.0]

    Rescaling one feature does change a distance-based learner, which is
    what makes the tree result non-trivial:

    >>> r2 = geron_tree_sensitivity_scale(X, y, a=0.001, b=0.0, feature=1)
    >>> r2["predictions_match"]
    True
    >>> r2["knn_predictions"], r2["knn_scaled_predictions"]
    ([0, 0, 0, 0], [0, 0, 0, 1])
    >>> r2["knn_match"]
    False

    References
    ----------
    Géron Ch 5
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_tree_sensitivity_scale: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    af, bf = float(a), float(b)
    if not np.isfinite(af) or af <= 0:
        raise ValueError(f"geron_tree_sensitivity_scale: a must be positive and finite, got {a!r} (a < 0 mirrors the axis)")
    if not np.isfinite(bf):
        raise ValueError(f"geron_tree_sensitivity_scale: b must be finite, got {b!r}")

    Xs = Xa.copy()
    if feature is None:
        cols = list(range(Xa.shape[1]))
    else:
        k = int(feature)
        if not (0 <= k < Xa.shape[1]):
            raise ValueError(f"geron_tree_sensitivity_scale: feature {k} out of range for {Xa.shape[1]} columns")
        cols = [k]
    Xs[:, cols] = af * Xa[:, cols] + bf

    base = geron_cart_algorithm(Xa, ya, criterion=criterion, max_depth=max_depth)
    scaled = geron_cart_algorithm(Xs, ya, criterion=criterion, max_depth=max_depth)

    def thresholds(node, out):
        if node["leaf"]:
            return out
        out.append((int(node["feature"]), float(node["threshold"])))
        thresholds(node["left"], out)
        thresholds(node["right"], out)
        return out

    t0 = thresholds(base["tree"], [])
    t1 = thresholds(scaled["tree"], [])
    expected = [(k, af * t + bf if k in cols else t) for k, t in t0]
    t_match = len(t0) == len(t1) and all(
        k0 == k1 and abs(e - t) < 1e-9 * max(1.0, abs(e)) for (k0, e), (k1, t) in zip(expected, t1)
    )
    p_match = list(base["predictions"]) == list(scaled["predictions"])

    def knn(Xtr):
        out = []
        for i in range(Xtr.shape[0]):
            d = np.sum((Xtr - Xtr[i]) ** 2, axis=1)
            d[i] = np.inf
            out.append(ya[int(np.argmin(d))])
        return out

    k0, k1 = knn(Xa), knn(Xs)

    return RichResult(
        title="Tree scale invariance",
        summary_lines=[("Predictions match", p_match), ("Thresholds match", t_match), ("1-NN match", list(k0) == list(k1))],
        interpretation="Axis-aligned splits depend only on the ordering of a feature, which an increasing affine map preserves.",
        payload={
            "predictions_match": bool(p_match),
            "predictions": list(base["predictions"]),
            "scaled_predictions": list(scaled["predictions"]),
            "thresholds": [t for _, t in t0],
            "scaled_thresholds": [t for _, t in t1],
            "expected_thresholds": [t for _, t in expected],
            "thresholds_match": bool(t_match),
            "knn_predictions": [v.item() if hasattr(v, "item") else v for v in k0],
            "knn_scaled_predictions": [v.item() if hasattr(v, "item") else v for v in k1],
            "knn_match": bool(list(k0) == list(k1)),
            "transform": {"a": af, "b": bf, "columns": cols},
            "estimate": 1.0 if p_match else 0.0,
            "n": int(Xa.shape[0]),
            "method": "affine rescaling experiment on CART vs a 1-NN control",
        },
    )


def cheatsheet():
    return "hmdtst: Decision trees are insensitive to feature scale (axis-aligned splits)"
