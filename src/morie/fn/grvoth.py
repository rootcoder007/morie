# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hard voting ensemble prediction (majority label among base classifiers)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_hard_voting"]

_METHOD = "Hard voting ensemble"


def geron_hard_voting(predictions):
    r"""Majority label across base classifiers.

    .. math::
        \hat y = \mathrm{mode}\bigl(h_1(x), h_2(x), \dots, h_L(x)\bigr)

    The law of large numbers is doing the work: L independent classifiers
    each right with probability :math:`p > 0.5` give a majority that is
    right with probability rising towards 1 in L.  "Independent" is the
    load-bearing word -- correlated learners vote the same way and the
    ensemble adds nothing, which is why bagging bothers to resample.
    Ties are broken towards the lowest class index, deterministically,
    and the tie count is reported because a tied ensemble is telling you
    something.

    Parameters
    ----------
    predictions : array-like, shape (L, m) or (L,)
        Integer class labels from L classifiers over m instances.

    Returns
    -------
    RichResult
        Payload keys ``y_hat``, ``vote_counts``, ``agreement`` (share of
        voters backing the winner), ``ties``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 6, Voting Classifier (hard) section.

    Examples
    --------
    Three classifiers, two instances: 2-1 for class 1, then 2-1 for
    class 0.

    >>> r = geron_hard_voting([[1, 0], [1, 1], [0, 0]])
    >>> r["y_hat"]
    [1, 0]
    >>> [round(a, 6) for a in r["agreement"]]
    [0.666667, 0.666667]

    A tie goes to the lower label and is flagged:

    >>> t = geron_hard_voting([[0], [1]])
    >>> t["y_hat"], t["ties"]
    ([0], 1)
    """
    P = np.atleast_2d(np.asarray(predictions))
    if P.ndim == 1:
        P = P[:, None]
    if P.ndim != 2 or P.size == 0:
        raise ValueError(f"predictions must be a non-empty (L, m) array, got shape {P.shape}.")
    if not np.all(P == np.round(np.asarray(P, dtype=float))):
        raise ValueError("hard voting needs integer class labels; pass probabilities to soft voting.")
    P = P.astype(int)
    if P.min() < 0:
        raise ValueError(f"class labels must be non-negative, got {int(P.min())}.")
    L, m = P.shape
    K = int(P.max()) + 1

    counts = np.zeros((m, K), dtype=int)
    for j in range(m):
        counts[j] = np.bincount(P[:, j], minlength=K)
    yhat = np.argmax(counts, axis=1)
    top = counts.max(axis=1)
    ties = int(np.sum((counts == top[:, None]).sum(axis=1) > 1))

    return RichResult(
        title="Hard voting",
        summary_lines=[("Voters", int(L)), ("Instances", int(m)), ("Ties", ties)],
        payload={
            "y_hat": yhat.astype(int).tolist(),
            "vote_counts": counts.tolist(),
            "agreement": (top / L).tolist(),
            "ties": ties,
            "estimate": yhat.astype(int).tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grvoth: y_hat = mode of the L predicted labels; ties to the lowest index and counted"
