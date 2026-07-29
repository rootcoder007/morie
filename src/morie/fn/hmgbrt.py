# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient boosted regression trees (GBRT): fit residual trees sequentially."""

import numpy as np

from ._richresult import RichResult
from .hmcart import geron_cart_algorithm, predict_tree

__all__ = ["geron_gradient_boosting"]


def geron_gradient_boosting(X, y, n_estimators=10, learning_rate=0.1, max_depth=2, loss="squared_error"):
    """
    Gradient boosted regression trees (GBRT): fit residual trees sequentially.

    Formula: F_{t+1}(x) = F_t(x) + eta * h_t(x); h_t fits residuals -grad L

    Boosting is done for real: the ensemble starts at the mean of ``y``
    (the constant that minimises squared loss), and at each round a tree
    is fitted to the negative gradient of the loss -- for squared error
    that is exactly the residual ``y - F(x)`` -- then added with weight
    ``learning_rate``. Tree fitting is DELEGATED to
    :func:`morie.fn.hmcart.geron_cart_algorithm` with ``criterion="mse"``.

    Shrinkage matters: with ``learning_rate = 1`` and deep enough trees
    the first round can already interpolate, and the rest add nothing. A
    small rate needs more rounds but generalises better, which is visible
    in ``loss_history`` -- it must be non-increasing, and that is checked
    rather than assumed.

    ``loss="absolute_error"`` fits the sign of the residual instead, which
    is the median-tracking variant.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Continuous targets.
    n_estimators : int, default 10
    learning_rate : float, default 0.1
        Shrinkage in (0, 1].
    max_depth : int, default 2
        Depth of each weak learner.
    loss : {"squared_error", "absolute_error"}, default "squared_error"

    Returns
    -------
    result : RichResult
        Keys: predictions, init, trees, loss_history, residual_history,
        train_mse, monotone, staged_predictions, estimate, n, method.

    Examples
    --------
    A step function with a stump: the first round moves each prediction a
    tenth of the way from the mean towards the target.

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> y = [0.0, 0.0, 10.0, 10.0]
    >>> r = geron_gradient_boosting(X, y, n_estimators=1, learning_rate=0.1, max_depth=1)
    >>> r["init"]
    5.0
    >>> [round(v, 6) for v in r["predictions"]]
    [4.5, 4.5, 5.5, 5.5]

    A full-rate round fits the residual exactly, so the loss drops to zero
    in one step:

    >>> r2 = geron_gradient_boosting(X, y, n_estimators=1, learning_rate=1.0, max_depth=1)
    >>> [round(v, 6) for v in r2["predictions"]]
    [0.0, 0.0, 10.0, 10.0]
    >>> r2["train_mse"]
    0.0

    The loss never increases:

    >>> r3 = geron_gradient_boosting(X, y, n_estimators=8, learning_rate=0.3, max_depth=1)
    >>> r3["monotone"]
    True
    >>> r3["loss_history"][0] > r3["loss_history"][-1]
    True

    References
    ----------
    Géron Ch 6
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    ya = np.asarray(y, dtype=float).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_gradient_boosting: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    if Xa.size == 0:
        raise ValueError("geron_gradient_boosting: X is empty")
    if not np.all(np.isfinite(Xa)) or not np.all(np.isfinite(ya)):
        raise ValueError("geron_gradient_boosting: X and y must be finite")
    B = int(n_estimators)
    if B < 1:
        raise ValueError(f"geron_gradient_boosting: n_estimators must be >= 1, got {n_estimators!r}")
    eta = float(learning_rate)
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"geron_gradient_boosting: learning_rate must lie in (0, 1], got {learning_rate!r}")
    if loss not in ("squared_error", "absolute_error"):
        raise ValueError(f"geron_gradient_boosting: loss must be 'squared_error' or 'absolute_error', got {loss!r}")
    md = int(max_depth)
    if md < 1:
        raise ValueError(f"geron_gradient_boosting: max_depth must be >= 1, got {max_depth!r}")

    init = float(np.mean(ya)) if loss == "squared_error" else float(np.median(ya))
    F = np.full(ya.shape, init)
    trees, hist, res_hist, staged = [], [], [], []

    def cur_loss(F):
        return float(np.mean((ya - F) ** 2)) if loss == "squared_error" else float(np.mean(np.abs(ya - F)))

    hist.append(cur_loss(F))
    for _ in range(B):
        resid = (ya - F) if loss == "squared_error" else np.sign(ya - F)
        res_hist.append(resid.tolist())
        if np.allclose(resid, 0):
            break
        t = geron_cart_algorithm(Xa, resid, criterion="mse", max_depth=md)["tree"]
        h = np.asarray(predict_tree(t, Xa), dtype=float)
        F = F + eta * h
        trees.append(t)
        staged.append(F.tolist())
        hist.append(cur_loss(F))

    mono = all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))

    return RichResult(
        title="Gradient boosted regression trees",
        summary_lines=[("Rounds", len(trees)), ("Train MSE", float(np.mean((ya - F) ** 2))), ("learning_rate", eta)],
        warnings=[] if mono else ["the loss increased during boosting, which should not happen for these losses"],
        interpretation="Each tree fits the negative gradient, so boosting is gradient descent in function space.",
        payload={
            "predictions": F.tolist(),
            "init": init,
            "trees": trees,
            "n_trees": int(len(trees)),
            "loss_history": hist,
            "residual_history": res_hist,
            "staged_predictions": staged,
            "train_mse": float(np.mean((ya - F) ** 2)),
            "train_mae": float(np.mean(np.abs(ya - F))),
            "monotone": bool(mono),
            "learning_rate": eta,
            "loss": loss,
            "estimate": float(np.mean((ya - F) ** 2)),
            "n": int(Xa.shape[0]),
            "method": "GBRT fitting trees to negative gradients; trees delegated to hmcart (criterion='mse')",
        },
    )


def cheatsheet():
    return "hmgbrt: Gradient boosted regression trees (GBRT): fit residual trees sequentially"
