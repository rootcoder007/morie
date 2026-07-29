# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""InfoNCE loss (van den Oord et al. 2018; Alammar Ch 10)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_infonce_loss"]


def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a zero vector has no direction; cosine "
                         "similarity with it is undefined.")
    return float(np.dot(a, b) / (na * nb))


def alammar_infonce_loss(anchor, positive, negatives, tau=0.07):
    """L = -log exp(s(a,p)/tau) / [exp(s(a,p)/tau) + sum exp(s(a,n)/tau)].

    References: Alammar and Grootendorst, Ch 10; van den Oord et al.
    (2018).
    """
    t = float(tau)
    if t <= 0:
        raise ValueError("the temperature must be positive.")
    a = np.atleast_1d(np.asarray(anchor, dtype=float))
    p = np.atleast_1d(np.asarray(positive, dtype=float))
    N = np.atleast_2d(np.asarray(negatives, dtype=float))
    sp = _cos(a, p) / t
    sn = np.array([_cos(a, N[i]) / t for i in range(N.shape[0])])
    zs = np.concatenate([[sp], sn])
    m = zs.max()
    logZ = m + np.log(np.exp(zs - m).sum())
    loss = float(logZ - sp)
    return RichResult(payload={
        "estimate": loss, "positive_similarity": sp * t,
        "negative_similarities": [float(v * t) for v in sn],
        "n": N.shape[0] + 1,
        "method": "InfoNCE (van den Oord et al. 2018)"})


def cheatsheet():
    return "alinfn: -log softmax over positive vs negatives at temperature tau"
