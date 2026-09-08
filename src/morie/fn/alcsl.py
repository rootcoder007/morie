# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cosine-similarity regression loss (Alammar Ch 10; SBERT)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_cosine_similarity_loss"]


def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a zero vector has no direction; cosine "
                         "similarity with it is undefined.")
    return float(np.dot(a, b) / (na * nb))


def alammar_cosine_similarity_loss(a, b, y_true):
    """L = (cos(a, b) - y_true)^2 per pair, mean over the batch.

    References: Alammar and Grootendorst, Ch 10; Reimers and Gurevych
    (2019).

    Examples
    --------
    >>> alammar_cosine_similarity_loss([[1.0, 0.0]], [[1.0, 0.0]],
    ...                                [1.0])["estimate"]
    0.0
    """
    A = np.atleast_2d(np.asarray(a, dtype=float))
    B = np.atleast_2d(np.asarray(b, dtype=float))
    y = np.atleast_1d(np.asarray(y_true, dtype=float))
    if A.shape != B.shape or A.shape[0] != len(y):
        raise ValueError("need matched pairs and one target per pair.")
    if np.any(np.abs(y) > 1):
        raise ValueError("targets are cosine values and must lie in "
                         "[-1, 1].")
    sims = np.array([_cos(A[i], B[i]) for i in range(A.shape[0])])
    losses = (sims - y) ** 2
    return RichResult(payload={
        "estimate": float(losses.mean()),
        "losses": [float(v) for v in losses],
        "similarities": [float(v) for v in sims], "n": len(y),
        "method": "Cosine similarity loss (Reimers and Gurevych 2019)"})


def cheatsheet():
    return "alcsl: mean (cos(a,b) - y)^2 over pairs"
