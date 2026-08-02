# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP contrastive image-text loss (symmetric InfoNCE over a batch)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_clip_contrastive_loss"]

_METHOD = "CLIP symmetric contrastive loss"


def _log_softmax_rows(Z):
    M = Z.max(axis=1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def geron_clip_contrastive_loss(image_embeddings, text_embeddings, tau=0.07,
                                normalize=True):
    r"""Symmetric InfoNCE over the matched pairs in a batch.

    .. math::
        L = \tfrac12\bigl[\mathrm{CE}(\mathrm{sim}(I,T)/\tau, \mathrm{diag})
          + \mathrm{CE}(\mathrm{sim}(T,I)/\tau, \mathrm{diag})\bigr]

    The batch supplies its own negatives: pair ``i``'s image must beat
    every *other* caption in the batch, and vice versa.  The loss
    therefore depends on batch size -- a batch of 2 has one negative per
    anchor and is close to trivial, which is why CLIP trained with
    batches in the tens of thousands.

    Parameters
    ----------
    image_embeddings, text_embeddings : array-like, shape (B, d)
        Paired embeddings; row ``i`` of each is a positive pair.
    tau : float, optional
        Temperature, positive. Default 0.07 (CLIP's learned value).
    normalize : bool, optional
        L2-normalise the embeddings first, making ``sim`` a cosine.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``loss_i2t``, ``loss_t2i``,
        ``similarity``, ``accuracy_i2t``, ``accuracy_t2i``,
        ``chance_loss``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 16, CLIP section.

    Examples
    --------
    Two orthogonal pairs at ``tau = 1``: each positive logit is 1, each
    negative 0, so the loss is ``log(1 + e^-1)``:

    >>> r = geron_clip_contrastive_loss([[1.0, 0.0], [0.0, 1.0]],
    ...                                 [[1.0, 0.0], [0.0, 1.0]], tau=1.0)
    >>> round(r["loss"], 6)
    0.313262
    >>> r["accuracy_i2t"]
    1.0

    Identical embeddings for every item give the chance loss ``log B``:

    >>> import math
    >>> r2 = geron_clip_contrastive_loss([[1.0, 0.0], [1.0, 0.0]],
    ...                                  [[1.0, 0.0], [1.0, 0.0]], tau=1.0)
    >>> round(r2["loss"], 6) == round(math.log(2), 6)
    True
    """
    I = np.atleast_2d(np.asarray(image_embeddings, dtype=float))
    T = np.atleast_2d(np.asarray(text_embeddings, dtype=float))
    if I.shape != T.shape:
        raise ValueError(
            f"image and text embeddings must have the same shape, got {I.shape} and {T.shape}."
        )
    if I.size == 0:
        raise ValueError("embeddings are empty.")
    if not np.all(np.isfinite(I)) or not np.all(np.isfinite(T)):
        raise ValueError("embeddings contain non-finite values.")
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be a positive finite float, got {tau}.")
    B = I.shape[0]

    if normalize:
        ni = np.linalg.norm(I, axis=1, keepdims=True)
        nt = np.linalg.norm(T, axis=1, keepdims=True)
        if np.any(ni == 0) or np.any(nt == 0):
            raise ValueError(
                "cannot L2-normalise a zero embedding; pass normalize=False or "
                "drop the zero rows."
            )
        I = I / ni
        T = T / nt

    sim = I @ T.T
    logits = sim / tau
    idx = np.arange(B)
    ls_i2t = _log_softmax_rows(logits)
    ls_t2i = _log_softmax_rows(logits.T)
    loss_i2t = float(-ls_i2t[idx, idx].mean())
    loss_t2i = float(-ls_t2i[idx, idx].mean())
    loss = 0.5 * (loss_i2t + loss_t2i)

    acc_i2t = float(np.mean(logits.argmax(axis=1) == idx))
    acc_t2i = float(np.mean(logits.argmax(axis=0) == idx))

    return RichResult(
        title="CLIP contrastive loss",
        summary_lines=[("Loss", loss), ("Batch size", B), ("Temperature", tau)],
        interpretation=(
            f"Chance loss for a batch of {B} is log(B) = {float(np.log(B)):.4f}; "
            "anything close to that means the embeddings carry no pairing signal."
        ),
        payload={
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "similarity": sim.tolist(),
            "accuracy_i2t": acc_i2t,
            "accuracy_t2i": acc_t2i,
            "chance_loss": float(np.log(B)),
            "tau": tau,
            "estimate": loss,
            "n": int(B),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grclp: CLIP loss = mean of image->text and text->image InfoNCE with diagonal targets"
