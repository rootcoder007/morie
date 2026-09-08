# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tree leaf prediction (leaf mean)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_tree_regression_leaf"]

_METHOD = "Regression tree leaf (mean of the leaf targets)"


def geron_tree_regression_leaf(y, leaf_mask=None):
    r"""Mean target of the instances that reach a leaf.

    .. math::
        \hat y_{\text{leaf}} = \frac{1}{|\text{leaf}|}
            \sum_{i \in \text{leaf}} y_i

    The mean is not an arbitrary choice: CART splits to minimise squared
    error, and the constant minimising squared error over a set *is* its
    mean, so the leaf value falls straight out of the split criterion.
    (Split on absolute error instead and the leaf would be the median.)
    The leaf MSE returned here is the same quantity the split search was
    weighing, which is why a leaf with one instance shows MSE 0 -- and
    why that is a warning, not an achievement.

    Note this is a *leaf* value, not a prediction over a whole dataset;
    for the split cost that produced the leaf see
    :mod:`morie.fn.grcart`.

    Parameters
    ----------
    y : array-like
        Target values.
    leaf_mask : array-like of bool, optional

    Returns
    -------
    RichResult
        Payload keys ``prediction``, ``mse``, ``n_leaf``, ``std``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 5, Regression Trees section.

    Examples
    --------
    >>> r = geron_tree_regression_leaf([1.0, 2.0, 6.0])
    >>> r["prediction"]
    3.0
    >>> round(r["mse"], 6)
    4.666667

    A single-instance leaf has zero MSE -- perfectly fitted, perfectly
    overfitted:

    >>> s = geron_tree_regression_leaf([1.0, 2.0, 6.0], [False, False, True])
    >>> s["prediction"], s["mse"], s["n_leaf"]
    (6.0, 0.0, 1)
    """
    yv = np.asarray(y, dtype=float).ravel()
    if yv.size == 0:
        raise ValueError("y is empty.")
    if not np.all(np.isfinite(yv)):
        raise ValueError("y contains non-finite values.")

    if leaf_mask is None:
        sel = yv
    else:
        mask = np.asarray(leaf_mask)
        if mask.shape != yv.shape:
            raise ValueError(f"leaf_mask has shape {mask.shape} but y has {yv.shape}.")
        sel = yv[mask.astype(bool)]
        if sel.size == 0:
            raise ValueError("leaf_mask selects no instances; an empty leaf has no prediction.")

    pred = float(sel.mean())
    mse = float(np.mean((sel - pred) ** 2))

    return RichResult(
        title="Regression tree leaf",
        summary_lines=[("Prediction", pred), ("Leaf size", int(sel.size)), ("Leaf MSE", mse)],
        payload={
            "prediction": pred,
            "mse": mse,
            "std": float(np.sqrt(mse)),
            "n_leaf": int(sel.size),
            "estimate": pred,
            "n": int(yv.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtrv: leaf value = mean of its targets (the squared-error minimiser); leaf MSE reported"
