# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BLIP: bootstrapped language-image pretraining."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_blip"]


def _l2_normalize(A):
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("geron_blip: an embedding has zero norm and cannot be projected onto the sphere")
    return A / norms


def _softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def geron_blip(images, texts, temperature=1.0, caption_logprobs=None):
    """
    BLIP: bootstrapped language-image pretraining.

    Formula: image-text contrastive + matching + captioning heads

    All three BLIP objectives on a batch of paired embeddings:

    * ITC -- symmetric InfoNCE over the cosine-similarity matrix scaled by
      1/temperature (image->text and text->image averaged);
    * ITM -- binary cross-entropy on the matched pairs against the hard
      negatives obtained by rotating the batch by one;
    * LM -- mean captioning cross-entropy, if per-token log-probabilities
      are supplied.

    Parameters
    ----------
    images : array-like, shape (n, d)
        Image embeddings.
    texts : array-like, shape (n, d)
        Text embeddings, row-aligned with `images`.
    temperature : float
        Positive softmax temperature.
    caption_logprobs : array-like, optional
        Per-caption mean token log-probabilities from a decoder, length n.

    Returns
    -------
    result : RichResult
        Keys: itc_loss, itm_loss, lm_loss, total_loss, similarity,
        retrieval_acc, estimate, n, method.

    Examples
    --------
    Two orthonormal pairs at temperature 1: each row's softmax is
    (e, 1)/(e+1), so the contrastive loss is log(1 + 1/e):

    >>> r = geron_blip([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], temperature=1.0)
    >>> round(float(r["itc_loss"]), 6)
    0.313262
    >>> float(r["retrieval_acc"])
    1.0

    Swapping the texts makes the pairing wrong and the loss larger:

    >>> r2 = geron_blip([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], temperature=1.0)
    >>> bool(r2["itc_loss"] > r["itc_loss"])
    True
    >>> float(r2["retrieval_acc"])
    0.0

    Lowering the temperature sharpens the softmax and shrinks the loss:

    >>> bool(geron_blip([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], temperature=0.1)["itc_loss"] < r["itc_loss"])
    True

    References
    ----------
    Géron Ch 16
    """
    I = np.asarray(images, dtype=float)
    T = np.asarray(texts, dtype=float)
    if I.ndim == 1:
        I = I.reshape(1, -1)
    if T.ndim == 1:
        T = T.reshape(1, -1)
    if I.ndim != 2 or T.ndim != 2:
        raise ValueError("geron_blip: images and texts must both be 2-D (n, d) embedding matrices")
    if I.shape != T.shape:
        raise ValueError(f"geron_blip: images has shape {I.shape} but texts has shape {T.shape}")
    n = I.shape[0]
    if n < 2:
        raise ValueError("geron_blip: contrastive learning needs at least 2 pairs in the batch")
    if not (np.all(np.isfinite(I)) and np.all(np.isfinite(T))):
        raise ValueError("geron_blip: embeddings must be finite")
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"geron_blip: temperature must be positive, got {tau}")

    In = _l2_normalize(I)
    Tn = _l2_normalize(T)
    sim = In @ Tn.T
    logits = sim / tau
    idx = np.arange(n)
    p_i2t = _softmax(logits, axis=1)
    p_t2i = _softmax(logits, axis=0)
    loss_i2t = float(-np.mean(np.log(np.clip(p_i2t[idx, idx], 1e-15, None))))
    loss_t2i = float(-np.mean(np.log(np.clip(p_t2i[idx, idx], 1e-15, None))))
    itc = 0.5 * (loss_i2t + loss_t2i)

    pos = sim[idx, idx]
    neg = sim[idx, (idx + 1) % n]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    itm = float(
        -np.mean(np.log(np.clip(sig(pos / tau), 1e-15, None)))
        - np.mean(np.log(np.clip(1.0 - sig(neg / tau), 1e-15, None)))
    ) / 2.0

    lm = None
    if caption_logprobs is not None:
        cl = np.asarray(caption_logprobs, dtype=float).ravel()
        if cl.size != n:
            raise ValueError(f"geron_blip: caption_logprobs has {cl.size} entries but the batch has {n} pairs")
        if np.any(cl > 0):
            raise ValueError("geron_blip: caption_logprobs must be log-probabilities (<= 0)")
        lm = float(-np.mean(cl))

    total = itc + itm + (lm or 0.0)
    retrieval = float(np.mean(np.argmax(sim, axis=1) == idx))

    return RichResult(
        title="BLIP pretraining objectives",
        summary_lines=[("ITC", itc), ("ITM", itm), ("LM", lm if lm is not None else "n/a"), ("Total", total)],
        payload={
            "itc_loss": itc,
            "itc_i2t": loss_i2t,
            "itc_t2i": loss_t2i,
            "itm_loss": itm,
            "lm_loss": lm,
            "total_loss": total,
            "similarity": sim,
            "retrieval_acc": retrieval,
            "temperature": tau,
            "estimate": total,
            "n": int(n),
            "method": "BLIP: contrastive + matching (+ captioning) objectives on paired embeddings",
        },
    )


def cheatsheet():
    return "hmblip: BLIP: bootstrapped language-image pretraining"
