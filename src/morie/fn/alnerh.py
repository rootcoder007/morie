# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-token NER head (Alammar Ch 4)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_ner_token_head"]


def alammar_ner_token_head(h_tokens, W, b, tags):
    """p(tag | h_t) = softmax(W h_t + b) for every token; CE loss when
    targets are supplied via ``tags`` (integer indices).

    References: Alammar and Grootendorst, Ch 4 (token classification).
    """
    H = np.atleast_2d(np.asarray(h_tokens, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if W.shape[1] != H.shape[1]:
        raise ValueError(
            f"W has {W.shape[1]} columns but hidden states have "
            f"{H.shape[1]} dimensions.")
    logits = H @ W.T + b
    z = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    pred = np.argmax(logits, axis=1)
    loss = None
    if tags is not None:
        t = np.atleast_1d(np.asarray(tags)).astype(int)
        if len(t) != H.shape[0]:
            raise ValueError(
                f"{len(t)} tags for {H.shape[0]} tokens.")
        if np.any((t < 0) | (t >= W.shape[0])):
            raise ValueError("tag index out of range.")
        loss = float(np.mean(-np.log(P[np.arange(len(t)), t])))
    return RichResult(payload={
        "probabilities": [[float(v) for v in r] for r in P],
        "predicted_tags": [int(v) for v in pred],
        "cross_entropy": loss,
        "estimate": float(loss) if loss is not None else float(pred[0]),
        "n": H.shape[0],
        "method": "Per-token NER head + CE (Alammar Ch 4)"})


def cheatsheet():
    return "alnerh: softmax(W h_t + b) per token, CE against tag targets"
