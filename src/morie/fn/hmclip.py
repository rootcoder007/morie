# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP: contrastive image-text pretraining."""

from . import _array_core as np

from ._richresult import RichResult
from .grclp import geron_clip_contrastive_loss

__all__ = ["geron_clip"]


def geron_clip(images, texts, tau=0.07, normalize=True, class_prompts=None):
    """
    CLIP: contrastive image-text pretraining.

    Formula: maximize cosine sim of matched (image, text); minimize for
    unmatched

    The symmetric InfoNCE objective is DELEGATED to
    :func:`morie.fn.grclp.geron_clip_contrastive_loss`. What this module
    adds is the downstream half of CLIP: zero-shot classification. Given
    ``class_prompts`` (one text embedding per class), each image is
    assigned the class whose prompt embedding has the highest cosine
    similarity -- the whole reason CLIP is trained this way.

    ``images`` and ``texts`` must be pre-computed embeddings of equal
    width; this module does not build the encoders.

    Parameters
    ----------
    images : array-like, shape (B, d)
        Image embeddings; row ``i`` matches row ``i`` of ``texts``.
    texts : array-like, shape (B, d)
        Text embeddings.
    tau : float, default 0.07
        Temperature (CLIP's learned value).
    normalize : bool, default True
        L2-normalise before taking similarities, making them cosines.
    class_prompts : array-like, shape (C, d), optional
        Prompt embeddings for zero-shot classification.

    Returns
    -------
    result : RichResult
        Keys: loss, loss_i2t, loss_t2i, similarity, accuracy_i2t,
        accuracy_t2i, matched_similarity, zero_shot, estimate, n, method.

    Examples
    --------
    Two orthogonal pairs at ``tau = 1``: every positive logit is 1 and
    every negative 0, so the loss is ``log(1 + e^-1)``:

    >>> r = geron_clip([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], tau=1.0)
    >>> round(r["loss"], 6)
    0.313262
    >>> r["accuracy_i2t"]
    1.0
    >>> [round(v, 6) for v in r["matched_similarity"]]
    [1.0, 1.0]

    Zero-shot: the first image matches the first prompt, the second the
    second:

    >>> r2 = geron_clip([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...                 class_prompts=[[1.0, 0.0], [0.0, 1.0]])
    >>> r2["zero_shot"]["predictions"]
    [0, 1]
    >>> round(r2["zero_shot"]["similarity"][0][0], 6)
    1.0

    References
    ----------
    Géron Ch 16
    """
    I = np.atleast_2d(np.asarray(images, dtype=float))
    T = np.atleast_2d(np.asarray(texts, dtype=float))
    if I.shape != T.shape:
        raise ValueError(f"geron_clip: images has shape {I.shape} but texts has shape {T.shape}; rows must be paired")
    if I.size == 0:
        raise ValueError("geron_clip: no embeddings supplied")

    base = geron_clip_contrastive_loss(I, T, tau=tau, normalize=normalize)
    S = np.asarray(base["similarity"], dtype=float)
    matched = np.diag(S).tolist()

    zs = None
    if class_prompts is not None:
        P = np.atleast_2d(np.asarray(class_prompts, dtype=float))
        if P.shape[1] != I.shape[1]:
            raise ValueError(f"geron_clip: class_prompts width {P.shape[1]} != embedding width {I.shape[1]}")
        if P.shape[0] == 0:
            raise ValueError("geron_clip: class_prompts is empty")
        if not np.all(np.isfinite(P)):
            raise ValueError("geron_clip: class_prompts contains non-finite values")
        In = I / np.linalg.norm(I, axis=1, keepdims=True) if normalize else I
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True) if normalize else P
        if normalize and (np.any(np.linalg.norm(I, axis=1) == 0) or np.any(np.linalg.norm(P, axis=1) == 0)):
            raise ValueError("geron_clip: cannot cosine-normalise a zero embedding")
        sim = In @ Pn.T
        zs = {
            "similarity": sim.tolist(),
            "predictions": sim.argmax(axis=1).astype(int).tolist(),
            "n_classes": int(P.shape[0]),
        }

    return RichResult(
        title="CLIP contrastive pretraining",
        summary_lines=[("Loss", float(base["loss"])), ("Batch", int(I.shape[0])), ("tau", float(tau))],
        interpretation="The batch's off-diagonal pairs are the negatives, so a bigger batch is a harder task.",
        payload={
            "loss": float(base["loss"]),
            "loss_i2t": float(base["loss_i2t"]),
            "loss_t2i": float(base["loss_t2i"]),
            "similarity": S.tolist(),
            "matched_similarity": matched,
            "accuracy_i2t": float(base["accuracy_i2t"]),
            "accuracy_t2i": float(base["accuracy_t2i"]),
            "chance_loss": float(base["chance_loss"]),
            "zero_shot": zs,
            "tau": float(tau),
            "estimate": float(base["loss"]),
            "n": int(I.shape[0]),
            "method": "CLIP symmetric InfoNCE (delegated to grclp) plus zero-shot prompt matching",
        },
    )


def cheatsheet():
    return "hmclip: CLIP: contrastive image-text pretraining"


# compact alias per ledger/NAMING.md
geronclip = geron_clip
