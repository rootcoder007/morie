# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""K-fold cross-validation: partition into k equal folds; rotate held-out fold."""

import numpy as np

from ._richresult import RichResult
from .grcvs import geron_cross_validation_score

__all__ = ["geron_kfold"]

_METHOD = "K-fold partition and cross-validated score"


def geron_kfold(X, y, k, seed=None, fit=None, predict=None, score=None):
    """
    K-fold cross-validation: partition into k equal folds; rotate held-out fold.

    Formula: avg CV score = (1/k) sum_i score(fold_i)

    This entry returns the *partition* -- the actual train and test index
    arrays for each fold -- and delegates the scoring to
    :func:`morie.fn.grcvs.geron_cross_validation_score`, whose fold
    construction it reproduces exactly (a seeded permutation split by
    ``array_split``), so the returned indices are the ones that produced
    the returned scores.

    Two invariants are checked rather than assumed: the test folds are
    disjoint, and together they cover every observation exactly once.
    ``array_split`` handles ``m`` not divisible by ``k`` by making the
    first ``m mod k`` folds one larger, so the fold sizes differ by at
    most one.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Targets.
    k : int
        Number of folds, ``2 <= k <= m``.
    seed : int, optional
        If given, rows are shuffled with this seed before splitting.
        Leave as None for contiguous folds.
    fit, predict, score : callable, optional
        Passed through to the delegate; the default is OLS scored by R^2.

    Returns
    -------
    result : RichResult
        Keys: cv_score, fold_scores, train_indices, test_indices,
        fold_sizes, estimate, n, method.

    Examples
    --------
    Noiseless ``y = 2x``: every fold recovers the slope, so every score
    is 1:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_kfold(X, [2.0, 4.0, 6.0, 8.0], k=2)
    >>> [round(s, 8) for s in r["fold_scores"]]
    [1.0, 1.0]
    >>> round(r["cv_score"], 8)
    1.0

    The partition is a partition: folds are disjoint and cover
    everything.

    >>> [list(map(int, t)) for t in r["test_indices"]]
    [[0, 1], [2, 3]]
    >>> [list(map(int, t)) for t in r["train_indices"]]
    [[2, 3], [0, 1]]

    Five points into two folds gives sizes 3 and 2:

    >>> u = geron_kfold([[1.0], [2.0], [3.0], [4.0], [5.0]],
    ...                 [2.0, 4.0, 6.0, 8.0, 10.0], k=2)
    >>> u["fold_sizes"]
    [3, 2]

    References
    ----------
    Géron Ch 2
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yy.size:
        raise ValueError(f"geron_kfold: X has {A.shape[0]} rows but y has {yy.size} entries")
    m = A.shape[0]
    K = int(k)
    if K < 2:
        raise ValueError(f"geron_kfold: k must be at least 2 folds, got {k!r}")
    if K > m:
        raise ValueError(f"geron_kfold: k={K} exceeds the {m} available observations")

    idx = np.arange(m)
    if seed is not None:
        idx = np.random.default_rng(int(seed)).permutation(m)
    folds = np.array_split(idx, K)
    test_idx = [np.asarray(f) for f in folds]
    train_idx = [np.setdiff1d(idx, f, assume_unique=False) for f in folds]

    covered = np.concatenate(test_idx)
    if np.unique(covered).size != m or covered.size != m:
        raise ValueError("geron_kfold: the folds do not partition the data; this is a bug in the split")

    inner = geron_cross_validation_score(
        A, yy, K=K, fit=fit, predict=predict, score=score,
        shuffle=seed is not None, random_state=None if seed is None else int(seed),
    )

    return RichResult(
        title=f"{K}-fold cross-validation",
        summary_lines=[
            ("Folds", K),
            ("CV score", float(inner["cv_score"])),
            ("SE across folds", float(inner["se"])),
            ("Worst fold", int(inner["worst_fold"])),
        ],
        interpretation=(
            "The spread across folds matters as much as the mean; one catastrophic fold usually "
            "means the folds are not exchangeable."
        ),
        payload={
            "cv_score": float(inner["cv_score"]),
            "fold_scores": inner["fold_scores"],
            "fold_sizes": inner["fold_sizes"],
            "se": float(inner["se"]),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "K": K,
            "estimate": float(inner["cv_score"]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkfd: K-fold partition (train/test indices) plus the CV score delegated to grcvs"
