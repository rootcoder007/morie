# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Triplet loss (Alammar Ch 10; SBERT)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_sbert_triplet_loss"]


def alammar_sbert_triplet_loss(anchor, positive, negative, margin=1.0):
    """L = max(0, d(a, p) - d(a, n) + margin), Euclidean d.

    An ACTIVE triplet (loss > 0) is the training signal; the payload
    says which triplets are active, since a batch of all-inactive
    triplets learns nothing however small its loss looks.

    References: Alammar and Grootendorst, Ch 10; Schroff et al. (2015).

    Examples
    --------
    >>> alammar_sbert_triplet_loss([[0.0]], [[0.0]], [[5.0]],
    ...                            margin=1.0)["estimate"]
    0.0
    """
    A = np.atleast_2d(np.asarray(anchor, dtype=float))
    P = np.atleast_2d(np.asarray(positive, dtype=float))
    N = np.atleast_2d(np.asarray(negative, dtype=float))
    m = float(margin)
    if not (A.shape == P.shape == N.shape):
        raise ValueError("anchor, positive and negative must align.")
    if m < 0:
        raise ValueError("margin must be non-negative.")
    dp = np.linalg.norm(A - P, axis=1)
    dn = np.linalg.norm(A - N, axis=1)
    losses = np.maximum(0.0, dp - dn + m)
    return RichResult(payload={
        "estimate": float(losses.mean()),
        "losses": [float(v) for v in losses],
        "active": [bool(v > 0) for v in losses],
        "d_positive": [float(v) for v in dp],
        "d_negative": [float(v) for v in dn], "n": A.shape[0],
        "method": "Triplet loss (Schroff et al. 2015)"})


def cheatsheet():
    return "altrip: max(0, d(a,p) - d(a,n) + margin), active flags reported"
