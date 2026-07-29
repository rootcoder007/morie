# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiple-negatives ranking loss (Henderson et al. 2017;
Alammar Ch 10)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_multiple_negatives_ranking"]


def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a zero vector has no direction; cosine "
                         "similarity with it is undefined.")
    return float(np.dot(a, b) / (na * nb))


def alammar_multiple_negatives_ranking(anchors, positives, tau=0.05):
    """In-batch negatives: every OTHER positive is a negative for a_i.

    L = -(1/B) sum_i log softmax_j(sim(a_i, p_j)/tau)[i].

    References: Alammar and Grootendorst, Ch 10; Henderson et al.
    (2017).
    """
    t = float(tau)
    if t <= 0:
        raise ValueError("the temperature must be positive.")
    A = np.atleast_2d(np.asarray(anchors, dtype=float))
    P = np.atleast_2d(np.asarray(positives, dtype=float))
    if A.shape != P.shape:
        raise ValueError("anchors and positives must align.")
    B = A.shape[0]
    if B < 2:
        raise ValueError(
            "in-batch negatives need a batch of at least 2; with one "
            "pair there are no negatives and the loss is trivially 0.")
    S = np.array([[_cos(A[i], P[j]) / t for j in range(B)]
                  for i in range(B)])
    Z = S - S.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    losses = -np.diag(logp)
    return RichResult(payload={
        "estimate": float(losses.mean()),
        "losses": [float(v) for v in losses],
        "similarity_matrix": [[float(v * t) for v in r] for r in S],
        "n": B,
        "method": "Multiple negatives ranking (Henderson et al. 2017)"})


def cheatsheet():
    return "almnrl: in-batch softmax over sim(a_i, p_j)/tau, diagonal is truth"
