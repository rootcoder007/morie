# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SimCSE unsupervised objective (Gao et al. 2021; Alammar Ch 10)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_simcse_dropout_aug"]


def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a zero vector has no direction; cosine "
                         "similarity with it is undefined.")
    return float(np.dot(a, b) / (na * nb))


def alammar_simcse_dropout_aug(embeddings_dropout1, embeddings_dropout2,
                               tau=0.05):
    """Two dropout passes of the SAME sentences are the positive pair;
    the rest of the batch are negatives.

    References: Alammar and Grootendorst, Ch 10; Gao et al. (2021).
    """
    t = float(tau)
    if t <= 0:
        raise ValueError("the temperature must be positive.")
    H1 = np.atleast_2d(np.asarray(embeddings_dropout1, dtype=float))
    H2 = np.atleast_2d(np.asarray(embeddings_dropout2, dtype=float))
    if H1.shape != H2.shape:
        raise ValueError("the two dropout passes must align.")
    B = H1.shape[0]
    if B < 2:
        raise ValueError("need a batch of at least 2 for in-batch "
                         "negatives.")
    S = np.array([[_cos(H1[i], H2[j]) / t for j in range(B)]
                  for i in range(B)])
    Z = S - S.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    losses = -np.diag(logp)
    return RichResult(payload={
        "estimate": float(losses.mean()),
        "losses": [float(v) for v in losses], "n": B,
        "method": "SimCSE with dropout augmentation (Gao et al. 2021)"})


def cheatsheet():
    return "alsmc: SimCSE -- dropout twins positive, batch mates negative"
