# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder-only transformer (BERT-family)."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdctr import block_params

__all__ = ["geron_encoder_only"]


def geron_encoder_only(
    X,
    n_layers=12,
    n_heads=12,
    d_model=768,
    vocab_size=30522,
    max_len=512,
    d_ff=None,
    n_segments=2,
    n_classes=None,
):
    """
    Encoder-only transformer (BERT-family).

    Formula: stacked self-attention + FFN; CLS token for classification

    An architecture specification, resolved against a concrete input as
    ``hmalex`` does. The block parameter count is DELEGATED to
    :func:`morie.fn.hmdctr.block_params`, since an encoder block and a
    decoder block are identical in size -- the difference is entirely the
    mask.

    That difference is the point, and it is made explicit here:
    ``attention_mask`` is all-``False`` (nothing forbidden) because every
    position may attend to every other, past and future. Bidirectionality
    is why an encoder cannot be trained by next-token prediction -- the
    answer is visible in the input -- and why BERT masks tokens instead.

    The special tokens are counted too: ``[CLS]`` is prepended and its
    final hidden state is the sequence representation, so the classifier
    head is ``d_model * n_classes`` and nothing else. ``cls_index`` is 0
    and ``seq_len`` includes the special tokens, which is where
    off-by-ones against ``max_len`` come from.

    Parameters
    ----------
    X : array-like
        Token ids of the raw sequence (before ``[CLS]``/``[SEP]``).
    n_layers, n_heads : int
    d_model : int, default 768
    vocab_size : int, default 30522
    max_len : int, default 512
    d_ff : int, optional
    n_segments : int, default 2
        Segment (sentence A/B) embedding rows.
    n_classes : int, optional
        Adds a linear classification head on the ``[CLS]`` state.

    Returns
    -------
    result : RichResult
        Keys: total_params, block_params, embedding_params, head_params,
        attention_mask, seq_len, cls_index, is_bidirectional, d_head,
        estimate, n, method.

    Examples
    --------
    A tiny configuration: the block cost matches the decoder's, since
    only the mask differs.

    >>> r = geron_encoder_only([1, 2, 3], n_layers=1, n_heads=2, d_model=4,
    ...                        vocab_size=10, max_len=8, n_segments=2)
    >>> r["block_params"]
    244
    >>> r["seq_len"], r["cls_index"]
    (5, 0)

    Every position may attend everywhere, so nothing is masked:

    >>> any(any(row) for row in r["attention_mask"])
    False
    >>> r["is_bidirectional"]
    True

    Embeddings are tokens, positions and segments; a two-class head adds
    ``4*2 + 2``:

    >>> r["embedding_params"]
    80
    >>> rc = geron_encoder_only([1, 2, 3], n_layers=1, n_heads=2, d_model=4,
    ...                         vocab_size=10, max_len=8, n_classes=2)
    >>> rc["head_params"]
    10
    >>> rc["total_params"] - r["total_params"]
    10

    References
    ----------
    Géron Ch 15
    """
    A = np.asarray(X)
    if A.ndim == 0 or A.size == 0:
        raise ValueError("geron_encoder_only: X must contain at least one token")
    raw = int(A.shape[-1])
    T = raw + 2  # [CLS] ... [SEP]
    L, Hh, d = int(n_layers), int(n_heads), int(d_model)
    V, M, S = int(vocab_size), int(max_len), int(n_segments)
    if L < 1:
        raise ValueError(f"geron_encoder_only: n_layers must be >= 1, got {n_layers!r}")
    if Hh < 1 or d < 1:
        raise ValueError("geron_encoder_only: n_heads and d_model must be >= 1")
    if d % Hh:
        raise ValueError(f"geron_encoder_only: d_model={d} is not divisible by n_heads={Hh}")
    if S < 1:
        raise ValueError(f"geron_encoder_only: n_segments must be >= 1, got {n_segments!r}")
    if T > M:
        raise ValueError(
            f"geron_encoder_only: sequence of {raw} tokens becomes {T} with [CLS] and [SEP], which exceeds max_len {M}"
        )

    per = block_params(d, d_ff=d_ff, cross_attention=False)
    emb = V * d + M * d + S * d
    head = 0 if n_classes is None else int(n_classes) * d + int(n_classes)
    if n_classes is not None and int(n_classes) < 2:
        raise ValueError(f"geron_encoder_only: n_classes must be >= 2, got {n_classes!r}")
    final_norm = 2 * d
    total = int(emb + L * per["total"] + final_norm + head)

    return RichResult(
        title="Encoder-only transformer",
        summary_lines=[("Layers", L), ("d_model", d), ("Total parameters", total)],
        interpretation="Bidirectional attention forbids next-token training, which is why BERT masks tokens instead.",
        payload={
            "total_params": total,
            "block_params": int(per["total"]),
            "per_block": per,
            "embedding_params": int(emb),
            "head_params": int(head),
            "attention_mask": np.zeros((T, T), dtype=bool).tolist(),
            "seq_len": int(T),
            "raw_len": int(raw),
            "cls_index": 0,
            "sep_index": int(T - 1),
            "is_bidirectional": True,
            "d_head": int(d // Hh),
            "n_layers": L,
            "n_heads": Hh,
            "d_model": d,
            "vocab_size": V,
            "max_len": M,
            "estimate": float(total),
            "n": int(L),
            "method": "encoder-only transformer resolved to exact parameter counts; block cost delegated to hmdctr",
        },
    )


def cheatsheet():
    return "hmencox: Encoder-only transformer (BERT-family)"
