# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BLIP image-text matching + contrastive + captioning objectives."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_blip_itm_itc"]

_METHOD = "BLIP multi-task objective (ITC + ITM + LM)"


def _log_softmax_rows(Z):
    M = Z.max(axis=-1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=-1, keepdims=True))


def _softplus(z):
    # log(1 + e^z), overflow-safe.
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def geron_blip_itm_itc(image_emb, text_emb, caption_logits, caption_targets,
                       tau=0.07, lam_itc=1.0, lam_itm=1.0, lam_lm=1.0,
                       normalize=True):
    r"""BLIP's three heads combined into one loss.

    .. math::
        L = \lambda_{\text{itc}}\,\mathrm{ITC}(I, T)
          + \lambda_{\text{itm}}\,\mathrm{ITM}(I, T)
          + \lambda_{\text{lm}}\,\mathrm{LM}(\text{caption} \mid \text{img})

    * **ITC** is symmetric InfoNCE over the batch (the CLIP objective) --
      alignment before fusion.
    * **ITM** is the binary matched/not-matched decision, scored here as
      cross-entropy over *all* :math:`B^2` pairs with the diagonal
      labelled 1 and everything else 0, the logit being
      :math:`\mathrm{sim}/\tau`.
    * **LM** is ordinary token cross-entropy on the caption.

    Understanding-and-generation in one model is BLIP's point: the ITC
    and ITM heads read images, the LM head writes about them, and the
    shared encoder has to serve both.

    Parameters
    ----------
    image_emb, text_emb : array-like, shape (B, d)
        Paired embeddings.
    caption_logits : array-like, shape (B, L, V)
        Per-token logits of the captioning head.
    caption_targets : array-like, shape (B, L)
        Target token indices; negatives are treated as padding and are
        excluded from the LM loss.
    tau : float, optional
        Temperature for ITC and ITM, positive. Default 0.07.
    lam_itc, lam_itm, lam_lm : float, optional
        Non-negative task weights.
    normalize : bool, optional
        L2-normalise embeddings so similarity is cosine.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``itc``, ``itm``, ``lm``, ``similarity``,
        ``itm_accuracy``, ``lm_perplexity``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 16, BLIP / BLIP-2 section.

    Examples
    --------
    A single matched pair at ``tau = 1``: ITC over a batch of one is
    zero, ITM is ``log(1 + e^-1)``, and uniform caption logits over two
    tokens give ``log 2``.

    >>> import math
    >>> r = geron_blip_itm_itc([[1.0, 0.0]], [[1.0, 0.0]],
    ...                        [[[0.0, 0.0]]], [[0]], tau=1.0)
    >>> round(abs(r["itc"]), 12)
    0.0
    >>> round(r["itm"], 6) == round(math.log(1 + math.exp(-1)), 6)
    True
    >>> round(r["lm"], 6) == round(math.log(2), 6)
    True
    >>> round(r["loss"], 6)
    1.006409
    >>> round(r["lm_perplexity"], 6)
    2.0
    """
    I = np.atleast_2d(np.asarray(image_emb, dtype=float))
    T = np.atleast_2d(np.asarray(text_emb, dtype=float))
    if I.shape != T.shape:
        raise ValueError(
            f"image_emb shape {I.shape} must match text_emb shape {T.shape}."
        )
    if I.size == 0:
        raise ValueError("embeddings are empty.")
    if not np.all(np.isfinite(I)) or not np.all(np.isfinite(T)):
        raise ValueError("embeddings contain non-finite values.")
    B = I.shape[0]

    CL = np.asarray(caption_logits, dtype=float)
    if CL.ndim != 3:
        raise ValueError(f"caption_logits must be 3-D (B, L, V), got ndim={CL.ndim}.")
    if CL.shape[0] != B:
        raise ValueError(
            f"caption_logits has batch {CL.shape[0]} but embeddings have {B}."
        )
    if not np.all(np.isfinite(CL)):
        raise ValueError("caption_logits contains non-finite values.")
    tgt = np.atleast_2d(np.asarray(caption_targets)).astype(int)
    if tgt.shape != CL.shape[:2]:
        raise ValueError(
            f"caption_targets must have shape {CL.shape[:2]}, got {tgt.shape}."
        )
    V = CL.shape[2]
    if np.any(tgt >= V):
        raise ValueError(f"caption target indices must be below the vocabulary size {V}.")

    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be a positive finite float, got {tau}.")
    for name, lam in (("lam_itc", lam_itc), ("lam_itm", lam_itm), ("lam_lm", lam_lm)):
        if not np.isfinite(float(lam)) or float(lam) < 0:
            raise ValueError(f"{name} must be a non-negative finite float, got {lam}.")
    lam_itc, lam_itm, lam_lm = float(lam_itc), float(lam_itm), float(lam_lm)

    if normalize:
        ni = np.linalg.norm(I, axis=1, keepdims=True)
        nt = np.linalg.norm(T, axis=1, keepdims=True)
        if np.any(ni == 0) or np.any(nt == 0):
            raise ValueError("cannot cosine-normalise a zero embedding.")
        I = I / ni
        T = T / nt

    sim = I @ T.T
    logits = sim / tau
    idx = np.arange(B)

    # ITC -- symmetric InfoNCE.
    itc = 0.5 * (
        float(-_log_softmax_rows(logits)[idx, idx].mean())
        + float(-_log_softmax_rows(logits.T)[idx, idx].mean())
    )

    # ITM -- binary cross-entropy over every pair, diagonal = match.
    labels = np.eye(B)
    itm = float(np.mean(_softplus(logits) - labels * logits))
    itm_acc = float(np.mean((logits > 0) == (labels > 0)))

    # LM -- token cross-entropy, padding (negative targets) ignored.
    mask = tgt >= 0
    if not mask.any():
        raise ValueError("every caption target is padding; the LM loss is undefined.")
    ls = _log_softmax_rows(CL)
    safe = np.where(mask, tgt, 0)
    tok_lp = np.take_along_axis(ls, safe[:, :, None], axis=2)[:, :, 0]
    lm = float(-np.sum(tok_lp * mask) / np.sum(mask))

    loss = lam_itc * itc + lam_itm * itm + lam_lm * lm

    return RichResult(
        title="BLIP multi-task loss",
        summary_lines=[("Loss", loss), ("ITC", itc), ("ITM", itm), ("LM", lm)],
        payload={
            "loss": loss,
            "itc": itc,
            "itm": itm,
            "lm": lm,
            "similarity": sim.tolist(),
            "itm_accuracy": itm_acc,
            "lm_perplexity": float(np.exp(lm)),
            "tau": tau,
            "weights": (lam_itc, lam_itm, lam_lm),
            "estimate": loss,
            "n": int(B),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grblip: BLIP loss = lam_itc*InfoNCE + lam_itm*pairwise BCE + lam_lm*caption CE"
