# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiclass one-vs-one."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_one_vs_one"]


def _centroid_pair(Xp, yp):
    """Default binary learner: nearest-centroid, labels 0/1."""
    c0 = Xp[yp == 0].mean(axis=0)
    c1 = Xp[yp == 1].mean(axis=0)

    def predict(A, _c0=c0, _c1=c1):
        B = np.atleast_2d(np.asarray(A, dtype=float))
        d0 = np.sum((B - _c0) ** 2, axis=1)
        d1 = np.sum((B - _c1) ** 2, axis=1)
        return (d1 <= d0).astype(float)

    return predict


def geron_one_vs_one(X, y, base_estimator=None, X_new=None):
    """
    Multiclass one-vs-one: train K(K-1)/2 pairwise binary classifiers.

    Formula: f_{i,j} for each pair (i<j); vote aggregation

    Each classifier sees only the two classes it arbitrates, so every
    training set is small -- K(K-1)/2 models but each on roughly 2m/K
    rows. For a learner whose cost grows worse than linearly in m (an
    SVM, in Geron's example) that is FASTER overall than one-vs-rest,
    which trains K models on all m rows each. Ties in the vote are broken
    toward the lowest class label, and ``tie_fraction`` reports how often
    that mattered.

    ``base_estimator(X_pair, y_pair)`` must return a predictor over the
    same columns emitting 0/1 for the (lower, higher) class of the pair.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Class labels; any values, sorted to fix the pair order.
    base_estimator : callable, optional
        Defaults to a nearest-centroid binary learner.
    X_new : array-like, optional
        Rows to classify; defaults to ``X``.

    Returns
    -------
    result : RichResult
        Keys: predict, predictions, classes, n_classifiers, votes,
        accuracy, tie_fraction, estimate, n, method.

    Examples
    --------
    Three well-separated classes in one dimension:

    >>> X = [[0.0], [1.0], [5.0], [6.0], [10.0], [11.0]]
    >>> r = geron_one_vs_one(X, [0, 0, 1, 1, 2, 2])
    >>> int(r["n_classifiers"]), float(r["accuracy"])
    (3, 1.0)
    >>> [int(p) for p in r["predict"]([[0.5], [10.5]])]
    [0, 2]

    K = 4 classes need 6 arbiters:

    >>> X4 = [[0.0], [3.0], [6.0], [9.0]]
    >>> int(geron_one_vs_one(X4, [0, 1, 2, 3])["n_classifiers"])
    6

    References
    ----------
    Geron Ch 3
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_one_vs_one: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.asarray(y).ravel()
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_one_vs_one: X has {A.shape[0]} rows but y has {yv.size} entries")
    classes = np.unique(yv)
    K = classes.size
    if K < 2:
        raise ValueError(f"geron_one_vs_one: need at least 2 classes, got {K}")
    est = _centroid_pair if base_estimator is None else base_estimator
    if not callable(est):
        raise ValueError("geron_one_vs_one: base_estimator must be callable")

    pairs, models = [], []
    for i in range(K):
        for j in range(i + 1, K):
            mask = (yv == classes[i]) | (yv == classes[j])
            yb = (yv[mask] == classes[j]).astype(float)
            f = est(A[mask], yb)
            if not callable(f):
                raise ValueError(f"geron_one_vs_one: base_estimator for pair {(i, j)} did not return a callable")
            pairs.append((i, j))
            models.append(f)

    def _vote(B):
        B = np.atleast_2d(np.asarray(B, dtype=float))
        tally = np.zeros((B.shape[0], K))
        for (i, j), f in zip(pairs, models):
            p = np.asarray(f(B), dtype=float).ravel()
            if p.size != B.shape[0]:
                raise ValueError(f"geron_one_vs_one: pair classifier {(i, j)} returned {p.size} predictions for {B.shape[0]} rows")
            tally[:, j] += p
            tally[:, i] += 1.0 - p
        return tally

    def predict(Xnew, _cls=classes, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        return _cls[np.argmax(_vote(B), axis=1)]

    target = A if X_new is None else np.atleast_2d(np.asarray(X_new, dtype=float))
    votes = _vote(target)
    pred = classes[np.argmax(votes, axis=1)]
    ties = float(np.mean(np.sum(votes == votes.max(axis=1, keepdims=True), axis=1) > 1))
    acc = float(np.mean(pred == yv)) if X_new is None else float("nan")

    return RichResult(
        title="One-vs-one multiclass",
        summary_lines=[("Classes", int(K)), ("Classifiers", len(models)), ("Training accuracy", acc)],
        interpretation="K(K-1)/2 small problems beat K big ones whenever the learner scales worse than linearly.",
        payload={
            "predict": predict,
            "predictions": pred,
            "classes": classes,
            "pairs": pairs,
            "n_classifiers": len(models),
            "votes": votes,
            "accuracy": acc,
            "tie_fraction": ties,
            "estimate": pred,
            "n": int(A.shape[0]),
            "method": "One-vs-one voting over K(K-1)/2 pairwise classifiers",
        },
    )


def cheatsheet():
    return "hmovo: One-vs-one multiclass voting"
