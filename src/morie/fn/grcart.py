# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CART split cost at a node."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_cart_split_cost"]

_METHOD = "CART split cost"


def _gini(y):
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(1.0 - np.sum(p * p))


def _entropy(y):
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(-np.sum(p * np.log2(p)))


def _mse(y):
    if y.size == 0:
        return 0.0
    return float(np.mean((y.astype(float) - y.astype(float).mean()) ** 2))


_CRITERIA = {"gini": _gini, "entropy": _entropy, "mse": _mse}


def geron_cart_split_cost(X, y, feature, threshold, criterion="gini"):
    r"""Cost of splitting a node on ``feature <= threshold``.

    .. math::
        J(k, t_k) = \frac{m_L}{m}G_L + \frac{m_R}{m}G_R

    CART is greedy and local: it minimises this one split's weighted
    impurity with no lookahead, which is why a tree can miss an XOR
    structure that two levels would capture trivially.  The impurity
    *reduction* relative to the parent is reported as well -- the cost on
    its own says nothing, since a node that was already pure has cost 0
    for every threshold.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Feature matrix. A 1-D array is treated as a single column.
    y : array-like, shape (m,)
        Labels (or targets, with ``criterion="mse"``).
    feature : int
        Column index to split on.
    threshold : float
        Instances with ``X[:, feature] <= threshold`` go left.
    criterion : {"gini", "entropy", "mse"}, optional
        Impurity measure.

    Returns
    -------
    RichResult
        Payload keys ``cost``, ``impurity_left``, ``impurity_right``,
        ``impurity_parent``, ``impurity_decrease``, ``n_left``,
        ``n_right``, ``estimate`` (the cost), ``n``, ``method``.

    References
    ----------
    Géron Ch 5, Eq 5-2 (CART cost function).

    Examples
    --------
    A threshold that separates the classes perfectly costs nothing:

    >>> r = geron_cart_split_cost([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1],
    ...                           feature=0, threshold=2.5)
    >>> r["cost"]
    0.0
    >>> round(r["impurity_decrease"], 6)
    0.5

    A worse threshold leaves the right child mixed:

    >>> r2 = geron_cart_split_cost([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1],
    ...                            feature=0, threshold=1.5)
    >>> round(r2["cost"], 6)
    0.333333
    >>> round(r2["impurity_right"], 6)
    0.444444
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"X must be a non-empty 2-D (m, n) array, got shape {X.shape}.")
    y = np.asarray(y).ravel()
    if y.size != X.shape[0]:
        raise ValueError(f"y has {y.size} entries but X has {X.shape[0]} rows.")
    feature = int(feature)
    if not (0 <= feature < X.shape[1]):
        raise ValueError(
            f"feature index {feature} out of range for {X.shape[1]} columns."
        )
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}.")
    if criterion not in _CRITERIA:
        raise ValueError(
            f"criterion must be one of {sorted(_CRITERIA)}, got {criterion!r}."
        )
    col = np.asarray(X[:, feature], dtype=float)
    if not np.all(np.isfinite(col)):
        raise ValueError(f"column {feature} of X contains non-finite values.")
    if criterion == "mse":
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values.")

    imp = _CRITERIA[criterion]
    m = y.size
    left = col <= threshold
    yl, yr = y[left], y[~left]
    ml, mr = yl.size, yr.size
    gl, gr = imp(yl), imp(yr)
    cost = (ml / m) * gl + (mr / m) * gr
    parent = imp(y)

    return RichResult(
        title="CART split cost",
        summary_lines=[("Cost", cost), ("Left / right", f"{ml} / {mr}")],
        warnings=[] if ml and mr else [
            f"threshold {threshold} sends every instance to one side; the split is degenerate."
        ],
        payload={
            "cost": float(cost),
            "impurity_left": gl,
            "impurity_right": gr,
            "impurity_parent": parent,
            "impurity_decrease": float(parent - cost),
            "n_left": int(ml),
            "n_right": int(mr),
            "criterion": criterion,
            "threshold": threshold,
            "feature": feature,
            "estimate": float(cost),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcart: CART split cost J = (m_L/m)G_L + (m_R/m)G_R, plus impurity decrease"
