# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-vs-One: K(K-1)/2 binary classifiers, majority vote."""

import numpy as np

from ._richresult import RichResult
from .grovr import train_logreg

__all__ = ["geron_one_vs_one"]

_METHOD = "One-vs-One multiclass reduction"


def geron_one_vs_one(X, y, base_fit=None, eta=0.5, n_iter=400):
    r"""Train a classifier for every pair of classes and let them vote.

    .. math::
        \hat y = \mathrm{majority\_vote}
            \bigl(\text{classifier}_{i,j}(x) : i < j\bigr)

    :math:`K(K-1)/2` models instead of OvR's :math:`K` -- but each is
    trained on only the two classes concerned, so for a learner whose
    cost is superlinear in the training-set size (SVMs, famously) OvO is
    the cheaper reduction despite the larger model count.  Each pair
    classifier is trained on its own subset, and every pair must be
    represented in the data, so a missing pair raises rather than
    quietly casting fewer votes.  Ties go to the lowest class index and
    are counted.  The binary learner is the deterministic logistic
    regression shared from :mod:`morie.fn.grovr`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like of int, shape (m,)
    base_fit : callable, optional
        ``base_fit(X_pair, y_pair) -> callable scorer``; positive score
        means the second class of the pair.
    eta, n_iter : float, int, optional

    Returns
    -------
    RichResult
        Payload keys ``predictions``, ``votes`` (m x K),
        ``n_classifiers``, ``pairs``, ``accuracy``, ``ties``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Multiclass (OvO) section.

    Examples
    --------
    Three separated clusters: ``3 * 2 / 2 = 3`` classifiers, and the
    winner takes both of its duels.

    >>> X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    >>> y = [0, 0, 1, 1, 2, 2]
    >>> r = geron_one_vs_one(X, y)
    >>> r["n_classifiers"]
    3
    >>> r["predictions"]
    [0, 0, 1, 1, 2, 2]
    >>> r["votes"][0]
    [2, 1, 0]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if yv.size != A.shape[0]:
        raise ValueError(f"y has {yv.size} labels but X has {A.shape[0]} rows.")
    if not np.all(yv == np.round(np.asarray(yv, dtype=float))):
        raise ValueError("y must hold integer class labels.")
    yv = yv.astype(int)
    classes = np.unique(yv)
    K = classes.size
    if K < 2:
        raise ValueError(f"OvO needs at least 2 classes, got {K}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")

    votes = np.zeros((A.shape[0], K), dtype=int)
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            ci, cj = classes[i], classes[j]
            sel = (yv == ci) | (yv == cj)
            if not sel.any():
                raise ValueError(f"no instances for the pair ({int(ci)}, {int(cj)}).")
            Xp = A[sel]
            yp = (yv[sel] == cj).astype(float)
            if yp.min() == yp.max():
                raise ValueError(
                    f"the pair ({int(ci)}, {int(cj)}) has only one class present; "
                    "the duel cannot be trained."
                )
            if base_fit is None:
                w = train_logreg(Xp, yp, eta=eta, n_iter=n_iter)
                s = np.hstack([np.ones((A.shape[0], 1)), A]) @ w
            else:
                if not callable(base_fit):
                    raise ValueError(f"base_fit must be callable, got {type(base_fit).__name__}.")
                model = base_fit(Xp, yp)
                if not callable(model):
                    raise ValueError("base_fit(X, y) must return a callable scorer.")
                s = np.asarray(model(A), dtype=float).ravel()
                if s.size != A.shape[0]:
                    raise ValueError(
                        f"pair ({int(ci)}, {int(cj)}) scorer returned {s.size} scores "
                        f"for {A.shape[0]} rows."
                    )
                if not np.all(np.isfinite(s)):
                    raise ValueError(f"pair ({int(ci)}, {int(cj)}) scorer returned non-finite scores.")
            votes[s > 0, j] += 1
            votes[s <= 0, i] += 1
            pairs.append((int(ci), int(cj)))

    pred = classes[np.argmax(votes, axis=1)]
    top = votes.max(axis=1)
    ties = int(np.sum((votes == top[:, None]).sum(axis=1) > 1))
    acc = float(np.mean(pred == yv))

    return RichResult(
        title="One-vs-One",
        summary_lines=[("Classifiers", len(pairs)), ("Training accuracy", acc), ("Ties", ties)],
        payload={
            "predictions": pred.astype(int).tolist(),
            "votes": votes.tolist(),
            "classes": classes.astype(int).tolist(),
            "pairs": pairs,
            "n_classifiers": len(pairs),
            "accuracy": acc,
            "ties": ties,
            "estimate": pred.astype(int).tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grovo: K(K-1)/2 pairwise duels, majority vote; each model sees only two classes' rows"
