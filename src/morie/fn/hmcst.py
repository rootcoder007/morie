# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contrastive learning: pull positives close, push negatives far."""

from . import _array_core as np

from ._richresult import RichResult
from .grctr import geron_contrastive_infonce

__all__ = ["geron_contrastive_learning"]


def geron_contrastive_learning(embeddings, positives, tau=0.1, normalize=True):
    """
    Contrastive learning: pull positives close, push negatives far.

    Formula: L = -log exp(sim(x,x+)/tau) / sum_j exp(sim(x,x_j)/tau)

    The loss itself is DELEGATED to
    :func:`morie.fn.grctr.geron_contrastive_infonce`. This module supplies
    the piece that formula leaves implicit: which rows are the negatives.
    ``positives[i]`` names the row that is the positive for anchor ``i``,
    and every *other* row of the batch (excluding the anchor and its own
    positive) becomes a negative for it -- the in-batch negative sampling
    that makes the objective cheap.

    A batch needs at least three rows for that to be non-empty, and an
    anchor may not be its own positive.

    Parameters
    ----------
    embeddings : array-like, shape (B, d)
        Batch of embeddings.
    positives : array-like of int, shape (B,)
        Index of the positive partner of each anchor.
    tau : float, default 0.1
        Temperature; smaller sharpens the softmax onto the hardest negative.
    normalize : bool, default True
        Use cosine similarity.

    Returns
    -------
    result : RichResult
        Keys: loss, per_anchor_loss, pos_sim, neg_sim, hardest_negative,
        accuracy, chance_loss, n_negatives, estimate, n, method.

    Examples
    --------
    Three unit vectors: rows 0 and 1 are a positive pair along ``x`` and
    row 2 is orthogonal, so at ``tau = 1`` each anchor sees one negative
    at cosine 0 and the loss is ``log(1 + e^-1)``:

    >>> emb = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_contrastive_learning(emb, [1, 0, 0], tau=1.0)
    >>> r["n_negatives"]
    1
    >>> round(r["per_anchor_loss"][0], 6)
    0.313262
    >>> round(r["pos_sim"][0], 6)
    1.0

    The third anchor's positive is orthogonal to it while its negative is
    orthogonal too, so it sits exactly at chance, ``log 2``:

    >>> import math
    >>> round(r["per_anchor_loss"][2], 9) == round(math.log(2), 9)
    True

    References
    ----------
    Géron Ch 16
    """
    E = np.atleast_2d(np.asarray(embeddings, dtype=float))
    if E.ndim != 2 or E.size == 0:
        raise ValueError(f"geron_contrastive_learning: embeddings must be a non-empty (B, d) array, got {E.shape}")
    if not np.all(np.isfinite(E)):
        raise ValueError("geron_contrastive_learning: embeddings contains non-finite values")
    B = E.shape[0]
    if B < 3:
        raise ValueError(
            f"geron_contrastive_learning: in-batch negatives need at least 3 embeddings, got {B}"
        )
    pos = np.asarray(positives).ravel()
    if pos.size != B:
        raise ValueError(f"geron_contrastive_learning: positives has {pos.size} entries but the batch has {B} rows")
    pi = pos.astype(int)
    if not np.array_equal(pi, pos):
        raise ValueError("geron_contrastive_learning: positives must be whole-number row indices")
    if pi.min() < 0 or pi.max() >= B:
        raise ValueError(f"geron_contrastive_learning: positive indices must lie in 0..{B - 1}")
    if np.any(pi == np.arange(B)):
        bad = np.flatnonzero(pi == np.arange(B)).tolist()
        raise ValueError(f"geron_contrastive_learning: anchors {bad} are their own positive")

    P = E[pi]
    neg_idx = [[j for j in range(B) if j != i and j != pi[i]] for i in range(B)]
    n_neg = len(neg_idx[0])
    if any(len(v) != n_neg for v in neg_idx):
        raise ValueError("geron_contrastive_learning: every anchor must have the same number of in-batch negatives")
    N = np.stack([E[v] for v in neg_idx], axis=0)

    base = geron_contrastive_infonce(E, P, N, tau=tau, normalize=normalize)

    return RichResult(
        title="Contrastive learning (in-batch negatives)",
        summary_lines=[("Loss", float(base["loss"])), ("Negatives per anchor", n_neg), ("tau", float(tau))],
        interpretation=f"Chance loss with {n_neg} negatives is log(1+N) = {float(np.log(1 + n_neg)):.4f}.",
        payload={
            "loss": float(base["loss"]),
            "per_anchor_loss": list(base["per_anchor_loss"]),
            "pos_sim": list(base["pos_sim"]),
            "neg_sim": base["neg_sim"],
            "hardest_negative": list(base["hardest_negative"]),
            "accuracy": float(base["accuracy"]),
            "chance_loss": float(base["chance_loss"]),
            "n_negatives": int(n_neg),
            "negative_indices": neg_idx,
            "tau": float(tau),
            "estimate": float(base["loss"]),
            "n": int(B),
            "method": "InfoNCE with in-batch negatives; loss delegated to grctr",
        },
    )


def cheatsheet():
    return "hmcst: Contrastive learning: pull positives close, push negatives far"
