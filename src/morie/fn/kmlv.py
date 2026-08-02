# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLaVA visual instruction tuning: project visual features into the
LLM's token space."""

from . import _array_core as np

from ._richresult import RichResult
from .kmclm import kamath_causal_lm_loss

__all__ = ["kamath_llava_visual_instruction"]


def kamath_llava_visual_instruction(image, W, visual_encoder, text_tokens,
                                    lm_head=None, targets=None,
                                    ignore_index=-100):
    """z_v = W ViT(image); inputs = [z_v; text_tokens]; causal-LM loss
    on the response.

    The projection is the whole trick: one linear map turns patch
    features into things the frozen LLM reads as tokens. The encoder
    is the caller's (``visual_encoder(image) -> (n_patches, d_v)``),
    and if ``lm_head`` and ``targets`` are supplied the response loss
    is DELEGATED to ``morie.fn.kmclm`` rather than re-derived here.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, visual instruction
    tuning (LLaVA).

    Examples
    --------
    >>> enc = lambda im: [[1.0, 2.0], [3.0, 4.0]]
    >>> W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> out = kamath_llava_visual_instruction("img", W, enc,
    ...                                       [[0.0, 0.0, 1.0]])
    >>> out["visual_tokens"]
    [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]]
    >>> out["n_visual"], out["n_text"]
    (2, 1)
    """
    if not callable(visual_encoder):
        raise ValueError(
            "visual_encoder must be callable image -> (n_patches, d_v).")
    feats = np.atleast_2d(np.asarray(visual_encoder(image), dtype=float))
    if feats.ndim != 2 or feats.size == 0:
        raise ValueError(
            "the visual encoder must return a non-empty "
            "(n_patches, d_v) array.")
    W = np.atleast_2d(np.asarray(W, dtype=float))
    d, d_v = W.shape
    if feats.shape[1] != d_v:
        raise ValueError(
            f"W maps {d_v}-dim visual features but the encoder returned "
            f"{feats.shape[1]}-dim ones.")
    txt = np.atleast_2d(np.asarray(text_tokens, dtype=float))
    if txt.shape[1] != d:
        raise ValueError(
            f"the projected visual tokens are {d}-dim but the text "
            f"embeddings are {txt.shape[1]}-dim; they must share the "
            "LLM's token space.")
    z_v = feats @ W.T
    inputs = np.vstack([z_v, txt])

    payload = {
        "visual_tokens": [[float(v) for v in row] for row in z_v],
        "inputs": [[float(v) for v in row] for row in inputs],
        "n_visual": int(z_v.shape[0]), "n_text": int(txt.shape[0]),
        "d_model": int(d),
        "estimate": float(z_v[0, 0]),
        "n": int(inputs.shape[0]),
        "method": "LLaVA visual instruction assembly (loss via kmclm)"}

    if (lm_head is None) != (targets is None):
        raise ValueError(
            "lm_head and targets must be supplied together; a head "
            "with nothing to predict has no loss.")
    if lm_head is not None:
        if not callable(lm_head):
            raise ValueError("lm_head must be callable inputs -> logits.")
        logits = np.asarray(lm_head(inputs), dtype=float)
        tgt = np.asarray(targets).ravel()
        if logits.ndim != 2 or logits.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"lm_head must return ({inputs.shape[0]}, V) logits; "
                f"got {logits.shape}.")
        if tgt.size != inputs.shape[0]:
            raise ValueError(
                f"targets must cover all {inputs.shape[0]} positions; "
                "mark the image and instruction positions with "
                f"ignore_index ({ignore_index}).")
        loss = kamath_causal_lm_loss(logits, tgt, ignore_index=ignore_index)
        payload["loss"] = float(loss["loss"])
        payload["perplexity"] = float(loss["perplexity"])
        payload["n_response_tokens"] = int(loss["n_tokens"])
        payload["estimate"] = float(loss["loss"])
    return RichResult(payload=payload)


def cheatsheet():
    return "kmlv: [W*ViT(image); text] soft tokens, response loss via kmclm"
