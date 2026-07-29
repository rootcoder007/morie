# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-captioning composition: project visual features into the
LM's space (Alammar Ch 9)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_image_captioning_pipeline"]


def alammar_image_captioning_pipeline(image, visual_encoder, projector,
                                      llm, prompt="Describe the image."):
    """z = W_proj VisEnc(img); caption = LLM([z; prompt]).

    Encoder and LLM are the caller's callables; the projection -- the
    piece the book's diagram actually specifies -- is computed here
    when ``projector`` is a matrix, and its dimensional contract is
    enforced.

    References: Alammar and Grootendorst, Ch 9 (LLaVA-style
    composition).
    """
    if not callable(visual_encoder) or not callable(llm):
        raise ValueError("visual_encoder and llm must be callable.")
    feats = np.atleast_1d(np.asarray(visual_encoder(image), dtype=float))
    if callable(projector):
        z = np.atleast_1d(np.asarray(projector(feats), dtype=float))
    else:
        W = np.atleast_2d(np.asarray(projector, dtype=float))
        if W.shape[1] != len(feats):
            raise ValueError(
                f"projector has {W.shape[1]} columns but the encoder "
                f"produced {len(feats)} features.")
        z = W @ feats
    caption = str(llm([float(v) for v in z], str(prompt)))
    return RichResult(payload={
        "caption": caption, "projected": [float(v) for v in z],
        "feature_dim": len(feats), "projected_dim": len(z),
        "estimate": float(len(caption)), "n": len(z),
        "method": "Visual projection into the LM (Alammar Ch 9)"})


def cheatsheet():
    return "alcap: caption = LLM([W VisEnc(img); prompt]), dims enforced"
