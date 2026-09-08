# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hard voting classifier: majority class vote among base models."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_voting_hard"]


def geron_voting_hard(models, X, y_true=None):
    """
    Hard voting classifier: majority class vote among base models.

    Formula: y_hat = mode_k({f_m(x)})

    Each base model is a callable ``m(X) -> labels`` returning one label
    per row. The ensemble label is the plurality vote, ties broken by the
    smallest label so the result is deterministic (a coin flip here would
    make the classifier irreproducible). Vote counts and per-model
    agreement with the ensemble are returned, because hard voting only
    beats its members when they are both accurate *and* uncorrelated.

    Parameters
    ----------
    models : sequence of callables
        At least one model; each maps X to labels of length n.
    X : array-like
        Rows to classify.
    y_true : array-like, optional
        Gold labels, for the accuracy comparison.

    Returns
    -------
    result : RichResult
        Keys: predicted, votes, member_predictions, member_accuracy,
        accuracy, agreement, estimate, n, method.

    Examples
    --------
    Three models: two vote class 1 and one votes class 0, so the majority
    carries even though the third model is right about the second row.

    >>> a = lambda X: [1, 1]
    >>> b = lambda X: [1, 0]
    >>> c = lambda X: [0, 0]
    >>> r = geron_voting_hard([a, b, c], [[0.0], [1.0]], y_true=[1, 0])
    >>> [int(v) for v in r["predicted"]]
    [1, 0]
    >>> float(r["accuracy"])
    1.0
    >>> [float(v) for v in r["member_accuracy"]]
    [0.5, 1.0, 0.5]

    Two of the three members are right only half the time, yet the vote
    is right on both rows -- that is the whole point of voting.

    References
    ----------
    Géron Ch 6
    """
    ms = list(models)
    if not ms:
        raise ValueError("geron_voting_hard: no base models supplied")
    for i, m in enumerate(ms):
        if not callable(m):
            raise ValueError(f"geron_voting_hard: model {i} is not callable; each model must map X to labels")
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    n = int(A.shape[0])
    if n == 0:
        raise ValueError("geron_voting_hard: X is empty")

    preds = []
    for i, m in enumerate(ms):
        p = np.asarray(m(X)).ravel()
        if p.size != n:
            raise ValueError(f"geron_voting_hard: model {i} returned {p.size} labels for {n} rows")
        preds.append(p)
    P = np.vstack(preds)
    classes = np.unique(P)

    votes = np.zeros((n, classes.size), dtype=int)
    for j, c in enumerate(classes):
        votes[:, j] = np.sum(P == c, axis=0)
    winner = np.argmax(votes, axis=1)  # np.argmax breaks ties towards the first (smallest) class
    pred = classes[winner]

    agreement = np.asarray([float(np.mean(p == pred)) for p in preds])
    acc = None
    member_acc = None
    if y_true is not None:
        g = np.asarray(y_true).ravel()
        if g.size != n:
            raise ValueError(f"geron_voting_hard: {n} rows but {g.size} gold labels")
        acc = float(np.mean(pred == g))
        member_acc = np.asarray([float(np.mean(p == g)) for p in preds])

    return RichResult(
        title="Hard voting ensemble",
        summary_lines=[
            ("Models", len(ms)),
            ("Rows", n),
            ("Classes", int(classes.size)),
            ("Ensemble accuracy", acc if acc is not None else "n/a (no labels)"),
        ],
        interpretation=(
            "Majority voting only helps when the members err on different rows; identical models vote "
            "identically and the ensemble is exactly as good as one of them."
        ),
        payload={
            "predicted": pred,
            "votes": votes,
            "classes": classes,
            "member_predictions": P,
            "member_accuracy": member_acc,
            "accuracy": acc,
            "agreement": agreement,
            "estimate": float(acc) if acc is not None else float(np.mean(np.max(votes, axis=1) / len(ms))),
            "n": n,
            "method": "Hard (plurality) voting with deterministic smallest-label tie-breaking",
        },
    )


def cheatsheet():
    return "hmvth: Hard voting classifier: majority class vote among base models"
