# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance-based learning: predict by measuring similarity to stored examples."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_instance_based"]

_METHOD = "k-nearest-neighbour (instance-based) prediction"


def geron_instance_based(X_train, y_train, x_query, k=1, task="auto", weights="uniform"):
    """
    Instance-based learning: predict by measuring similarity to stored examples.

    Formula: y_hat(x) = aggregate over k nearest neighbors by distance d(x, x_i)

    The counterpart to model-based learning
    (:func:`morie.fn.hmmod.geron_model_based`): nothing is fitted, the
    training set *is* the model, and all the work happens at prediction
    time.  Training cost zero, prediction cost ``O(m n)`` per query, and
    memory proportional to the whole dataset.

    Distances are Euclidean and unscaled, which is the trap: a feature
    measured in metres and one in millimetres contribute a thousandfold
    differently. The per-feature spread of the training data is returned
    so an unscaled input announces itself instead of quietly dominating
    every neighbourhood.

    ``task="auto"`` picks classification when the targets are integral
    with few distinct values, regression otherwise; pass it explicitly
    when the guess would be wrong.

    Parameters
    ----------
    X_train : array-like, shape (m, n)
        Stored instances.
    y_train : array-like, shape (m,)
        Stored targets or labels.
    x_query : array-like, shape (n,) or (q, n)
        Query point(s).
    k : int
        Neighbours to consult, ``1 <= k <= m``.
    task : {"auto", "classification", "regression"}
        Aggregation rule.
    weights : {"uniform", "distance"}
        Uniform vote/mean, or inverse-distance weighting.

    Returns
    -------
    result : RichResult
        Keys: prediction, neighbors, distances, task, feature_ranges,
        estimate, n, method.

    Examples
    --------
    One neighbour, so the prediction is the nearest stored label:

    >>> X = [[0.0], [1.0], [10.0]]
    >>> r = geron_instance_based(X, [0, 0, 1], [0.9], k=1)
    >>> int(r["prediction"][0])
    0

    Three neighbours vote, and 2-of-3 wins:

    >>> v = geron_instance_based(X, [0, 1, 1], [5.0], k=3)
    >>> int(v["prediction"][0])
    1

    Regression averages the neighbours: the mean of 2 and 4 is 3.

    >>> g = geron_instance_based([[0.0], [1.0], [9.0]], [2.0, 4.0, 100.0],
    ...                          [0.5], k=2, task="regression")
    >>> float(g["prediction"][0])
    3.0

    Inverse-distance weighting moves it toward the closer neighbour.
    At query 0.25 the distances are 0.25 and 0.75, so the weights are
    4 and 4/3 and the prediction is ``(4*2 + (4/3)*4)/(4 + 4/3) = 2.5``:

    >>> w = geron_instance_based([[0.0], [1.0]], [2.0, 4.0], [0.25], k=2,
    ...                          task="regression", weights="distance")
    >>> round(float(w["prediction"][0]), 9)
    2.5

    The distances come back so the neighbourhood can be inspected:

    >>> [round(float(d), 6) for d in r["distances"][0]]
    [0.1]

    References
    ----------
    Géron Ch 1
    """
    A = np.asarray(X_train, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_instance_based: X_train must be a non-empty 2-D array, got shape {A.shape}")
    yy = np.asarray(y_train).ravel()
    if yy.size != A.shape[0]:
        raise ValueError(f"geron_instance_based: X_train has {A.shape[0]} rows but y_train has {yy.size} entries")
    Q = np.asarray(x_query, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if Q.ndim != 2 or Q.size == 0:
        raise ValueError(f"geron_instance_based: x_query must be 1-D or 2-D, got shape {Q.shape}")
    if Q.shape[1] != A.shape[1]:
        raise ValueError(f"geron_instance_based: x_query has {Q.shape[1]} features but X_train has {A.shape[1]}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Q)):
        raise ValueError("geron_instance_based: X_train and x_query must be finite")
    kk = int(k)
    if not (1 <= kk <= A.shape[0]):
        raise ValueError(f"geron_instance_based: k must lie in 1..{A.shape[0]}, got {k!r}")
    if weights not in ("uniform", "distance"):
        raise ValueError(f"geron_instance_based: weights must be 'uniform' or 'distance', got {weights!r}")
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"geron_instance_based: task must be 'auto', 'classification' or 'regression', got {task!r}")

    if task == "auto":
        numeric = np.issubdtype(yy.dtype, np.number)
        integral = numeric and np.all(np.asarray(yy, dtype=float) == np.floor(np.asarray(yy, dtype=float)))
        resolved = "classification" if (not numeric or (integral and np.unique(yy).size <= max(2, yy.size // 2))) else "regression"
    else:
        resolved = task

    D = np.sqrt(np.clip(np.sum((Q[:, None, :] - A[None, :, :]) ** 2, axis=2), 0.0, None))
    nn = np.argsort(D, axis=1, kind="mergesort")[:, :kk]
    dist = np.take_along_axis(D, nn, axis=1)

    preds = []
    for q in range(Q.shape[0]):
        idx = nn[q]
        d = dist[q]
        if weights == "distance":
            if np.any(d == 0):
                w = (d == 0).astype(float)
            else:
                w = 1.0 / d
        else:
            w = np.ones(kk)
        if resolved == "regression":
            vals = np.asarray(yy[idx], dtype=float)
            preds.append(float(np.sum(w * vals) / np.sum(w)))
        else:
            classes = np.unique(yy[idx])
            totals = [float(np.sum(w[yy[idx] == c])) for c in classes]
            preds.append(classes[int(np.argmax(totals))])

    pred = np.asarray(preds)
    ranges = A.max(axis=0) - A.min(axis=0)
    spread = float(np.max(ranges) / np.min(ranges[ranges > 0])) if np.any(ranges > 0) else 1.0

    return RichResult(
        title="Instance-based prediction",
        summary_lines=[
            ("Stored instances", int(A.shape[0])),
            ("Neighbours", kk),
            ("Task", resolved),
            ("Feature-range ratio", spread),
        ],
        warnings=(
            [
                f"the widest feature range is {spread:.4g} times the narrowest; unscaled Euclidean "
                f"distance will be dominated by the wide one."
            ]
            if spread > 10
            else []
        ),
        interpretation=(
            "Nothing is fitted -- the training set is the model, so all the cost is at prediction time."
        ),
        payload={
            "prediction": pred,
            "neighbors": nn,
            "distances": dist,
            "task": resolved,
            "feature_ranges": ranges,
            "estimate": float(pred[0]) if np.issubdtype(pred.dtype, np.number) else 0.0,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmins: instance-based kNN -- no fitting, all cost at query time, unscaled-feature warning"
