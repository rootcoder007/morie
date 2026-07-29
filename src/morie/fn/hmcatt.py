# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-attention: Q from decoder, K/V from encoder."""

import numpy as np

from ._richresult import RichResult
from .grca import geron_cross_attention as _grca

__all__ = ["geron_cross_attention"]


def geron_cross_attention(dec_h, enc_h, W_Q, W_K, W_V, mask=None):
    """
    Cross-attention: Q from the decoder, K/V from the encoder.

    Formula: Att(dec_h W_Q, enc_h W_K, enc_h W_V)

    The scaled dot-product itself is DELEGATED to
    :func:`morie.fn.grca.geron_cross_attention`, which already implements
    this exact formula including the ``1/sqrt(d_k)`` scaling and the
    ``-inf`` masking. This module keeps Géron's argument names and adds
    the per-query diagnostics that make an attention map readable: the
    entropy of each query's distribution (in bits) and which encoder
    position it attends to most.

    Uniform attention over ``Te`` positions has entropy ``log2(Te)`` bits;
    an entropy near 0 means a query has collapsed onto one source token.

    Parameters
    ----------
    dec_h : array-like, shape (Td, d_model)
        Decoder hidden states (the queries).
    enc_h : array-like, shape (Te, d_model)
        Encoder outputs (the keys and values).
    W_Q, W_K : array-like, shape (d_model, d_k)
    W_V : array-like, shape (d_model, d_v)
    mask : array-like of bool, shape (Td, Te), optional
        True marks a disallowed (query, source) pair.

    Returns
    -------
    result : RichResult
        Keys: context, output, attention_weights, logits, d_k, entropy,
        argmax, max_entropy, estimate, n, method.

    Examples
    --------
    A zero query attends uniformly, so the context is the mean value
    vector and the entropy is 1 bit over two source positions:

    >>> r = geron_cross_attention([[0.0]], [[1.0], [3.0]], [[0.0]], [[1.0]], [[1.0]])
    >>> [round(w, 6) for w in r["attention_weights"][0]]
    [0.5, 0.5]
    >>> [round(v, 6) for v in r["context"][0]]
    [2.0]
    >>> round(r["entropy"][0], 6)
    1.0

    Masking the second source sends all the weight to the first, and the
    entropy drops to zero:

    >>> r2 = geron_cross_attention([[0.0]], [[1.0], [3.0]], [[0.0]], [[1.0]], [[1.0]],
    ...                            mask=[[False, True]])
    >>> [round(v, 6) for v in r2["context"][0]]
    [1.0]
    >>> round(r2["entropy"][0], 6)
    0.0
    >>> r2["argmax"]
    [0]

    References
    ----------
    Géron Ch 15
    """
    base = _grca(dec_h, enc_h, W_Q, W_K, W_V, mask=mask)
    A = np.asarray(base["attention_weights"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(A > 0, A * np.log2(np.where(A > 0, A, 1.0)), 0.0)
    ent = -terms.sum(axis=1) + 0.0
    ent = np.where(ent == 0, 0.0, ent)
    out = np.asarray(base["output"], dtype=float)

    return RichResult(
        title="Cross-attention",
        summary_lines=[("Decoder length", int(A.shape[0])), ("Encoder length", int(A.shape[1])), ("d_k", int(base["d_k"]))],
        interpretation="Queries come from the decoder, keys and values from the encoder, so no causal mask is needed.",
        payload={
            "context": out.tolist(),
            "output": out.tolist(),
            "attention_weights": A.tolist(),
            "logits": base["logits"],
            "scale": float(base["scale"]),
            "d_k": int(base["d_k"]),
            "entropy": ent.tolist(),
            "max_entropy": float(np.log2(A.shape[1])),
            "argmax": A.argmax(axis=1).astype(int).tolist(),
            "estimate": float(out.mean()),
            "n": int(A.shape[0]),
            "method": "scaled dot-product cross-attention (delegated to grca) with per-query entropy",
        },
    )


def cheatsheet():
    return "hmcatt: Cross-attention: Q from decoder, K/V from encoder"
