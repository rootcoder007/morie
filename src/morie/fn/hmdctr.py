# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decoder-only transformer (GPT family)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_decoder_only", "block_params", "causal_mask"]


def block_params(d_model, d_ff=None, cross_attention=False):
    """Exact parameter count of one transformer block, itemised.

    Attention is ``4 d^2 + 4 d`` (Q, K, V and the output projection, each
    with a bias), the feed-forward network ``2 d d_ff + d_ff + d``, and
    each LayerNorm ``2 d``. A cross-attention sub-layer adds another
    attention block and another LayerNorm.
    """
    d = int(d_model)
    if d < 1:
        raise ValueError(f"block_params: d_model must be >= 1, got {d_model!r}")
    ff = 4 * d if d_ff is None else int(d_ff)
    if ff < 1:
        raise ValueError(f"block_params: d_ff must be >= 1, got {d_ff!r}")
    attn = 4 * d * d + 4 * d
    ffn = 2 * d * ff + ff + d
    norms = 2 * (2 * d)
    out = {"self_attention": attn, "ffn": ffn, "layer_norms": norms}
    if cross_attention:
        out["cross_attention"] = attn
        out["layer_norms"] = norms + 2 * d
    out["total"] = int(sum(v for k, v in out.items() if k != "total"))
    return out


def causal_mask(n):
    """Boolean mask, True where attention is FORBIDDEN (strictly future)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"causal_mask: n must be >= 1, got {n}")
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def geron_decoder_only(X, n_layers=12, n_heads=12, d_model=768, vocab_size=50257, max_len=1024, d_ff=None, tie_embeddings=True):
    """
    Decoder-only transformer (GPT family).

    Formula: causal self-attention; predict next token

    An architecture specification, resolved against a concrete input in
    the ``hmalex`` manner: every parameter block is counted exactly and
    the causal mask is materialised for the sequence actually supplied.

    Two constraints are enforced rather than assumed. ``d_model`` must be
    divisible by ``n_heads`` -- heads partition the model width, they do
    not each get a full copy -- and the sequence may not exceed
    ``max_len``, since the learned positional table has no entry beyond
    it.

    The parameter budget is dominated by the blocks, ``12 d^2`` each
    (``4 d^2`` attention plus ``8 d^2`` feed-forward), which is the
    back-of-envelope every model card is built on. With
    ``tie_embeddings`` the output projection reuses the token embedding
    matrix, saving ``vocab * d`` parameters -- the single largest saving
    available in a small model.

    Parameters
    ----------
    X : array-like
        Token ids, shape ``(T,)`` or ``(B, T)``; used for its length.
    n_layers, n_heads : int
    d_model : int, default 768
    vocab_size : int, default 50257
    max_len : int, default 1024
    d_ff : int, optional
        Default ``4 * d_model``.
    tie_embeddings : bool, default True

    Returns
    -------
    result : RichResult
        Keys: total_params, block_params, embedding_params, per_block,
        d_head, mask, seq_len, n_layers, n_heads, flops_per_token,
        estimate, n, method.

    Examples
    --------
    A tiny configuration, counted exactly: one block at ``d = 4`` holds
    ``4*16+16`` attention, ``2*4*16+16+4`` FFN and ``2*2*4`` norm
    parameters.

    >>> r = geron_decoder_only([1, 2, 3], n_layers=1, n_heads=2, d_model=4,
    ...                        vocab_size=10, max_len=8)
    >>> r["per_block"]["self_attention"], r["per_block"]["ffn"], r["per_block"]["layer_norms"]
    (80, 148, 16)
    >>> r["block_params"]
    244
    >>> r["d_head"]
    2

    Embeddings are tokens plus positions, and tying saves the output
    matrix:

    >>> r["embedding_params"]
    72
    >>> untied = geron_decoder_only([1, 2, 3], n_layers=1, n_heads=2, d_model=4,
    ...                             vocab_size=10, max_len=8, tie_embeddings=False)
    >>> untied["total_params"] - r["total_params"]
    50

    The causal mask forbids exactly the strict upper triangle:

    >>> r["mask"]
    [[False, True, True], [False, False, True], [False, False, False]]
    >>> r["seq_len"]
    3

    A width that does not divide evenly among the heads is an error:

    >>> geron_decoder_only([1], n_heads=5, d_model=768)
    Traceback (most recent call last):
      ...
    ValueError: geron_decoder_only: d_model=768 is not divisible by n_heads=5

    References
    ----------
    Géron Ch 15
    """
    A = np.asarray(X)
    if A.ndim == 0 or A.size == 0:
        raise ValueError("geron_decoder_only: X must contain at least one token")
    T = int(A.shape[-1])
    L = int(n_layers)
    Hh = int(n_heads)
    d = int(d_model)
    V = int(vocab_size)
    M = int(max_len)
    if L < 1:
        raise ValueError(f"geron_decoder_only: n_layers must be >= 1, got {n_layers!r}")
    if Hh < 1:
        raise ValueError(f"geron_decoder_only: n_heads must be >= 1, got {n_heads!r}")
    if d < 1:
        raise ValueError(f"geron_decoder_only: d_model must be >= 1, got {d_model!r}")
    if d % Hh:
        raise ValueError(f"geron_decoder_only: d_model={d} is not divisible by n_heads={Hh}")
    if V < 1:
        raise ValueError(f"geron_decoder_only: vocab_size must be >= 1, got {vocab_size!r}")
    if M < 1:
        raise ValueError(f"geron_decoder_only: max_len must be >= 1, got {max_len!r}")
    if T > M:
        raise ValueError(f"geron_decoder_only: sequence length {T} exceeds max_len {M}; there is no positional entry for it")

    per = block_params(d, d_ff=d_ff, cross_attention=False)
    emb = V * d + M * d
    head = 0 if tie_embeddings else V * d + V
    final_norm = 2 * d
    total = int(emb + L * per["total"] + final_norm + head)

    return RichResult(
        title="Decoder-only transformer",
        summary_lines=[("Layers", L), ("d_model", d), ("Total parameters", total)],
        tables=[{
            "title": "Parameter budget",
            "headers": ["part", "params"],
            "rows": [["embeddings", emb], ["blocks", L * per["total"]], ["final norm", final_norm], ["output head", head]],
        }],
        interpretation="Each block costs about 12 d^2 parameters; the causal mask is what makes next-token training parallel.",
        payload={
            "total_params": total,
            "block_params": int(per["total"]),
            "per_block": per,
            "embedding_params": int(emb),
            "output_head_params": int(head),
            "d_head": int(d // Hh),
            "mask": causal_mask(T).tolist(),
            "seq_len": T,
            "n_layers": L,
            "n_heads": Hh,
            "d_model": d,
            "d_ff": 4 * d if d_ff is None else int(d_ff),
            "vocab_size": V,
            "max_len": M,
            "tie_embeddings": bool(tie_embeddings),
            "flops_per_token": int(2 * total),
            "estimate": float(total),
            "n": int(L),
            "method": "decoder-only transformer resolved to exact parameter counts and a causal mask",
        },
    )


def cheatsheet():
    return "hmdctr: Decoder-only transformer (GPT family)"
