# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder-decoder transformer (original architecture)."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdctr import block_params, causal_mask

__all__ = ["geron_encoder_decoder_transformer"]


def geron_encoder_decoder_transformer(
    src,
    tgt,
    n_layers=6,
    n_heads=8,
    d_model=512,
    vocab_size=37000,
    max_len=512,
    d_ff=2048,
    share_embeddings=True,
):
    """
    Encoder-decoder transformer (original architecture).

    Formula: encoder stack + decoder stack with cross-attention

    The 2017 architecture, resolved against concrete ``src`` and ``tgt``
    sequences in the ``hmalex`` manner. Block costs are DELEGATED to
    :func:`morie.fn.hmdctr.block_params`; the only structural difference
    is that a decoder block carries a third sub-layer, so it costs one
    extra attention block and one extra LayerNorm -- reported as
    ``extra_per_decoder_block``.

    Three masks are produced, and they are not interchangeable:

    * ``src_mask``: none, the encoder is bidirectional;
    * ``tgt_mask``: causal, so the decoder cannot read ahead of the token
      it is predicting;
    * ``cross_mask``: none over source positions, because the whole
      source is available when generating any target token -- this is why
      cross-attention needs no causal structure at all.

    The base configuration of the paper (``N=6``, ``d=512``, ``h=8``,
    ``d_ff=2048``) is the default, so the defaults reproduce the model
    the formula line refers to.

    Parameters
    ----------
    src, tgt : array-like
        Source and target token ids.
    n_layers : int, default 6
        Layers in each stack.
    n_heads : int, default 8
    d_model : int, default 512
    vocab_size : int, default 37000
    max_len : int, default 512
    d_ff : int, default 2048
    share_embeddings : bool, default True
        Share one embedding matrix between the two stacks.

    Returns
    -------
    result : RichResult
        Keys: total_params, encoder_params, decoder_params,
        embedding_params, src_mask, tgt_mask, cross_mask,
        extra_per_decoder_block, d_head, estimate, n, method.

    Examples
    --------
    A tiny configuration: the encoder block costs 244 parameters at
    ``d = 4`` and the decoder block adds another attention plus a norm.

    >>> r = geron_encoder_decoder_transformer([1, 2], [3, 4, 5], n_layers=1,
    ...     n_heads=2, d_model=4, vocab_size=10, max_len=8, d_ff=16)
    >>> r["encoder_block_params"], r["decoder_block_params"]
    (244, 332)
    >>> r["extra_per_decoder_block"]
    88

    The target mask is causal over 3 positions; the cross mask is empty
    over the 2 source positions:

    >>> r["tgt_mask"]
    [[False, True, True], [False, False, True], [False, False, False]]
    >>> r["cross_mask"]
    [[False, False], [False, False], [False, False]]
    >>> any(any(row) for row in r["src_mask"])
    False

    Sharing the embedding matrix saves exactly one ``vocab * d``:

    >>> untied = geron_encoder_decoder_transformer([1, 2], [3, 4, 5], n_layers=1,
    ...     n_heads=2, d_model=4, vocab_size=10, max_len=8, d_ff=16,
    ...     share_embeddings=False)
    >>> untied["total_params"] - r["total_params"]
    40

    References
    ----------
    Géron Ch 15
    """
    S = np.asarray(src)
    Tg = np.asarray(tgt)
    if S.ndim == 0 or S.size == 0 or Tg.ndim == 0 or Tg.size == 0:
        raise ValueError("geron_encoder_decoder_transformer: src and tgt must each contain at least one token")
    Ts, Tt = int(S.shape[-1]), int(Tg.shape[-1])
    L, Hh, d = int(n_layers), int(n_heads), int(d_model)
    V, M = int(vocab_size), int(max_len)
    if L < 1 or Hh < 1 or d < 1:
        raise ValueError("geron_encoder_decoder_transformer: n_layers, n_heads and d_model must be >= 1")
    if d % Hh:
        raise ValueError(f"geron_encoder_decoder_transformer: d_model={d} is not divisible by n_heads={Hh}")
    if Ts > M or Tt > M:
        raise ValueError(
            f"geron_encoder_decoder_transformer: sequence lengths ({Ts}, {Tt}) exceed max_len {M}"
        )

    enc = block_params(d, d_ff=d_ff, cross_attention=False)
    dec = block_params(d, d_ff=d_ff, cross_attention=True)
    emb = V * d + M * d if share_embeddings else 2 * V * d + M * d
    out_head = V * d + V
    norms = 2 * (2 * d)
    total = int(emb + L * enc["total"] + L * dec["total"] + norms + out_head)

    return RichResult(
        title="Encoder-decoder transformer",
        summary_lines=[("Layers per stack", L), ("d_model", d), ("Total parameters", total)],
        interpretation="Only the decoder is causal; cross-attention sees the whole source, which is why it needs no mask.",
        payload={
            "total_params": total,
            "encoder_params": int(L * enc["total"]),
            "decoder_params": int(L * dec["total"]),
            "encoder_block_params": int(enc["total"]),
            "decoder_block_params": int(dec["total"]),
            "extra_per_decoder_block": int(dec["total"] - enc["total"]),
            "embedding_params": int(emb),
            "output_head_params": int(out_head),
            "src_mask": np.zeros((Ts, Ts), dtype=bool).tolist(),
            "tgt_mask": causal_mask(Tt).tolist(),
            "cross_mask": np.zeros((Tt, Ts), dtype=bool).tolist(),
            "src_len": Ts,
            "tgt_len": Tt,
            "d_head": int(d // Hh),
            "n_layers": L,
            "n_heads": Hh,
            "d_model": d,
            "d_ff": int(d_ff),
            "share_embeddings": bool(share_embeddings),
            "estimate": float(total),
            "n": int(2 * L),
            "method": "original encoder-decoder transformer resolved to exact parameter counts and its three masks",
        },
    )


def cheatsheet():
    return "hmencd: Encoder-decoder transformer (original architecture)"
