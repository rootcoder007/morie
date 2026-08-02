# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mistral-7B: open-weights 7B-parameter decoder-only LLM."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mistral7b"]

_METHOD = "Mistral-7B architecture accounting"


def geron_mistral7b(
    prompt,
    n_tokens,
    n_layers=32,
    d_model=4096,
    n_heads=32,
    n_kv_heads=8,
    d_ff=14336,
    vocab_size=32000,
    window=4096,
    dtype_bytes=2,
):
    """
    Mistral-7B: open-weights 7B-parameter decoder-only LLM.

    Formula: decoder-only transformer with sliding-window attention

    An architecture entry, resolved against a concrete prompt.  Two
    departures from the GPT-3 shape carry the whole design, and both are
    computed here rather than described:

    **Grouped-query attention.**  32 query heads share 8 key/value
    heads, so ``W_k`` and ``W_v`` are a quarter the width of ``W_q``.
    The parameter saving is modest; the *cache* saving is the point --
    the KV cache is 4x smaller, and the cache is what limits how many
    sequences fit on a device.

    **Sliding-window attention.**  Each token attends to the previous
    ``window`` tokens only, so the attention mask is banded and the
    per-token attention cost stops growing with the sequence.
    Information still travels further than one window because each layer
    shifts the receptive field: after ``n_layers`` layers the effective
    span is ``window * n_layers`` tokens.

    The SwiGLU feed-forward block uses three matrices (gate, up, down),
    not two, which is why ``d_ff`` is 14336 rather than ``4 * d_model``.

    Parameters
    ----------
    prompt : array-like of int
        Token ids.
    n_tokens : int
        Tokens to generate (non-negative).
    n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab_size, window : int
        Architecture; defaults are the published 7B configuration.
    dtype_bytes : int
        Bytes per cached element.

    Returns
    -------
    result : RichResult
        Keys: total_parameters, parameters_per_layer, breakdown, d_head,
        kv_cache_bytes, kv_cache_saving, attention_mask,
        effective_context, estimate, n, method.

    Examples
    --------
    The default configuration totals 7.24 billion parameters:

    >>> r = geron_mistral7b([1, 2, 3, 4], n_tokens=2)
    >>> r["total_parameters"]
    7241732096
    >>> r["d_head"]
    128

    Per layer: attention ``4096*4096*2 + 4096*1024*2 = 41943040``,
    SwiGLU ``3 * 4096 * 14336 = 176160768``, two RMSNorms ``2*4096``:

    >>> r["parameters_per_layer"]
    218112000
    >>> r["breakdown"]["attention"], r["breakdown"]["feedforward"]
    (1342177280, 5637144576)

    Grouped-query attention makes the KV cache four times smaller than
    multi-head attention would:

    >>> round(r["kv_cache_saving"], 6)
    4.0

    Sliding-window attention gives a banded causal mask.  With a window
    of 2 over 4 tokens, token 3 sees tokens 2 and 3 only:

    >>> w = geron_mistral7b([1, 2, 3, 4], n_tokens=0, window=2)
    >>> [[int(v) for v in row] for row in w["attention_mask"]]
    [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]

    Depth extends the reach: 32 layers of a 4096 window span 131072
    tokens:

    >>> r["effective_context"]
    131072

    References
    ----------
    Géron Ch 15
    """
    ids = np.atleast_1d(np.asarray(prompt)).ravel()
    if ids.size == 0:
        raise ValueError("geron_mistral7b: prompt is empty")
    if not np.issubdtype(ids.dtype, np.integer):
        f = np.asarray(ids, dtype=float)
        if not np.all(f == np.floor(f)):
            raise ValueError("geron_mistral7b: prompt must contain integer token ids")
        ids = f.astype(np.int64)
    n_layers, d_model, n_heads = int(n_layers), int(d_model), int(n_heads)
    n_kv_heads, d_ff, vocab_size = int(n_kv_heads), int(d_ff), int(vocab_size)
    win, db = int(window), int(dtype_bytes)
    n_new = int(n_tokens)
    for name, v in (
        ("n_layers", n_layers), ("d_model", d_model), ("n_heads", n_heads),
        ("n_kv_heads", n_kv_heads), ("d_ff", d_ff), ("vocab_size", vocab_size),
        ("window", win), ("dtype_bytes", db),
    ):
        if v < 1:
            raise ValueError(f"geron_mistral7b: {name} must be a positive integer, got {v}")
    if n_new < 0:
        raise ValueError(f"geron_mistral7b: n_tokens must be non-negative, got {n_new}")
    if d_model % n_heads:
        raise ValueError(f"geron_mistral7b: d_model={d_model} is not divisible by n_heads={n_heads}")
    if n_heads % n_kv_heads:
        raise ValueError(
            f"geron_mistral7b: n_heads={n_heads} is not divisible by n_kv_heads={n_kv_heads}; "
            f"grouped-query attention needs whole groups"
        )
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise ValueError(
            f"geron_mistral7b: token ids must lie in 0..{vocab_size - 1}, "
            f"got range {int(ids.min())}..{int(ids.max())}"
        )

    d_head = d_model // n_heads
    d_kv = n_kv_heads * d_head
    attn_per_layer = 2 * d_model * d_model + 2 * d_model * d_kv
    ffn_per_layer = 3 * d_model * d_ff
    norm_per_layer = 2 * d_model
    per_layer = attn_per_layer + ffn_per_layer + norm_per_layer

    breakdown = {
        "token_embedding": vocab_size * d_model,
        "output_head": vocab_size * d_model,
        "attention": n_layers * attn_per_layer,
        "feedforward": n_layers * ffn_per_layer,
        "norms": n_layers * norm_per_layer + d_model,
    }
    total = int(sum(breakdown.values()))

    n_prompt = int(ids.size)
    total_len = n_prompt + n_new
    kv_bytes = 2 * n_layers * total_len * d_kv * db
    kv_bytes_mha = 2 * n_layers * total_len * d_model * db

    i = np.arange(n_prompt)[:, None]
    j = np.arange(n_prompt)[None, :]
    mask = (j <= i) & (j > i - win)

    return RichResult(
        title="Mistral-7B architecture",
        summary_lines=[
            ("Total parameters", total),
            ("Layers x d_model", f"{n_layers} x {d_model}"),
            ("Query / KV heads", f"{n_heads} / {n_kv_heads}"),
            ("Sliding window", win),
            ("KV cache (bytes)", int(kv_bytes)),
        ],
        interpretation=(
            "Grouped-query attention shrinks the cache, not the compute; the sliding window bounds "
            "per-token attention cost while depth still carries information window*n_layers tokens."
        ),
        payload={
            "total_parameters": total,
            "parameters_per_layer": int(per_layer),
            "breakdown": breakdown,
            "d_head": int(d_head),
            "d_kv": int(d_kv),
            "kv_cache_bytes": int(kv_bytes),
            "kv_cache_bytes_mha": int(kv_bytes_mha),
            "kv_cache_saving": float(kv_bytes_mha) / float(kv_bytes),
            "attention_mask": mask,
            "window": win,
            "effective_context": int(win * n_layers),
            "n_prompt_tokens": n_prompt,
            "n_generated": n_new,
            "estimate": float(total),
            "n": n_prompt,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmis7: Mistral-7B accounting -- GQA cache saving, SwiGLU width, banded sliding-window mask"
