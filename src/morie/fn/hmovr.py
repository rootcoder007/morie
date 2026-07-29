# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiclass one-vs-rest."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_one_vs_rest"]


def _centroid_score(Xb, yb):
    """Default binary scorer: signed distance to the midpoint between centroids."""
    c1 = Xb[yb == 1].mean(axis=0)
    c0 = Xb[yb == 0].mean(axis=0) if np.any(yb == 0) else c1
    w = c1 - c0
    nw = np.linalg.norm(w)
    w = w / nw if nw > 0 else w
    b = -float(w @ (c0 + c1) / 2.0)

    def score(A, _w=w, _b=b):
        B = np.atleast_2d(np.asarray(A, dtype=float))
        return B @ _w + _b

    return score


def geron_one_vs_rest(X, y, base_estimator=None, X_new=None):
    """
    Multiclass one-vs-rest (OvR): train K binary classifiers.

    Formula: for k in 1..K: f_k distinguishes class k vs all others

    K models, each trained on ALL the rows, and the winner is the one
    with the highest score. Two consequences follow and are reported:
    the per-classifier training sets are heavily imbalanced (one class
    against K-1, so ``positive_rate`` falls as 1/K), and the scores must
    be COMPARABLE across classifiers for the argmax to mean anything --
    a decision function that is well calibrated within a model can still
    be on a different scale from its neighbours. ``margin`` is the gap
    between the winning and runner-up scores; small margins are where
    that comparability assumption bites.

    ``base_estimator(X, y_binary)`` must return a callable emitting a
    real-valued score (higher = more likely that class).

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    base_estimator : callable, optional
        Defaults to a centroid-based linear discriminant score.
    X_new : array-like, optional
        Rows to classify; defaults to ``X``.

    Returns
    -------
    result : RichResult
        Keys: predict, predictions, classes, n_classifiers, scores,
        margin, positive_rate, accuracy, estimate, n, method.

    Examples
    --------
    >>> X = [[0.0], [1.0], [5.0], [6.0], [10.0], [11.0]]
    >>> r = geron_one_vs_rest(X, [0, 0, 1, 1, 2, 2])
    >>> int(r["n_classifiers"]), float(r["accuracy"])
    (3, 1.0)
    >>> [int(p) for p in r["predict"]([[0.5], [10.5]])]
    [0, 2]

    Each binary problem is 1-against-2, so a third of the rows are
    positive:

    >>> [round(float(p), 6) for p in r["positive_rate"]]
    [0.333333, 0.333333, 0.333333]

    References
    ----------
    Geron Ch 3
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_one_vs_rest: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.asarray(y).ravel()
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_one_vs_rest: X has {A.shape[0]} rows but y has {yv.size} entries")
    classes = np.unique(yv)
    K = classes.size
    if K < 2:
        raise ValueError(f"geron_one_vs_rest: need at least 2 classes, got {K}")
    est = _centroid_score if base_estimator is None else base_estimator
    if not callable(est):
        raise ValueError("geron_one_vs_rest: base_estimator must be callable")

    models = []
    rates = []
    for k in range(K):
        yb = (yv == classes[k]).astype(float)
        rates.append(float(yb.mean()))
        f = est(A, yb)
        if not callable(f):
            raise ValueError(f"geron_one_vs_rest: base_estimator for class {classes[k]!r} did not return a callable")
        models.append(f)

    def _scores(B):
        B = np.atleast_2d(np.asarray(B, dtype=float))
        S = np.empty((B.shape[0], K))
        for k, f in enumerate(models):
            s = np.asarray(f(B), dtype=float).ravel()
            if s.size != B.shape[0]:
                raise ValueError(f"geron_one_vs_rest: classifier {k} returned {s.size} scores for {B.shape[0]} rows")
            S[:, k] = s
        return S

    def predict(Xnew, _cls=classes, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return _cls[np.argmax(_scores(B), axis=1)]

    target = A if X_new is None else np.atleast_2d(np.asarray(X_new, dtype=float))
    S = _scores(target)
    pred = classes[np.argmax(S, axis=1)]
    part = np.sort(S, axis=1)
    margin = part[:, -1] - part[:, -2]
    acc = float(np.mean(pred == yv)) if X_new is None else float("nan")

    return RichResult(
        title="One-vs-rest multiclass",
        summary_lines=[("Classes", int(K)), ("Classifiers", K), ("Training accuracy", acc)],
        interpretation="The argmax across models assumes their scores are on one scale; small margins test that.",
        payload={
            "predict": predict,
            "predictions": pred,
            "classes": classes,
            "n_classifiers": int(K),
            "scores": S,
            "margin": margin,
            "positive_rate": rates,
            "accuracy": acc,
            "estimate": pred,
            "n": int(A.shape[0]),
            "method": "One-vs-rest argmax over K binary decision functions",
        },
    )


def cheatsheet():
    return "hmovr: One-vs-rest multiclass argmax"
