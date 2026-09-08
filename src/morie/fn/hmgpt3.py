# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPT-3: 175B-parameter autoregressive LM capable of in-context learning."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gpt3"]

_METHOD = "GPT-3 decoder-only architecture accounting"


def geron_gpt3(
    prompt,
    n_tokens,
    n_layers=96,
    d_model=12288,
    n_heads=96,
    d_ff=None,
    vocab_size=50257,
    n_ctx=2048,
    dtype_bytes=2,
):
    """
    GPT-3: 175B-parameter autoregressive LM capable of in-context learning.

    Formula: same decoder-only architecture, massive scale

    This is an architecture-only entry, so it resolves the architecture
    against a concrete input rather than pretending to run 175 billion
    weights.  Given the published GPT-3 175B configuration (96 layers,
    d_model 12288, 96 heads, d_ff = 4*d_model, 50257 BPE tokens, a 2048
    token window) it returns the exact parameter count layer by layer,
    the shape of every tensor a decode step produces for *this* prompt,
    the KV-cache footprint of generating ``n_tokens`` more, and the
    forward FLOPs implied by the usual ``2N`` per-token rule.

    The token embedding is tied to the output head, as in GPT-3, so the
    unembedding contributes no separate parameters.

    Parameters
    ----------
    prompt : array-like of int
        Token ids.  Every id must lie in ``0 .. vocab_size-1`` and the
        prompt plus the requested continuation must fit in ``n_ctx``.
    n_tokens : int
        Number of tokens to generate (non-negative).
    n_layers, d_model, n_heads, vocab_size, n_ctx : int
        Architecture; the defaults are the 175B configuration.
    d_ff : int, optional
        Feed-forward width; defaults to ``4 * d_model``.
    dtype_bytes : int
        Bytes per stored activation, for the cache accounting (2 = fp16).

    Returns
    -------
    result : RichResult
        Keys: total_parameters, parameters_per_layer, breakdown,
        shape_trace, d_head, kv_cache_bytes, flops_per_token,
        estimate, n, method.

    Examples
    --------
    The default configuration is the published 175B model:

    >>> r = geron_gpt3([1, 2, 3], n_tokens=5)
    >>> r["total_parameters"]
    174604259328
    >>> r["d_head"]
    128

    Per layer: ``12*d^2`` weights plus ``13*d`` biases and LayerNorm
    parameters, which for d_model=12288 is 1811939328 + 159744:

    >>> r["parameters_per_layer"]
    1812099072
    >>> r["breakdown"]["token_embedding"]
    617558016

    A two-layer toy of the same shape family, checked by hand:
    ``12*4^2 + 13*4 = 244`` per layer, embedding ``10*4 = 40``,
    positions ``8*4 = 32``, final LayerNorm ``2*4 = 8``:

    >>> t = geron_gpt3([0, 1], n_tokens=1, n_layers=2, d_model=4, n_heads=2,
    ...                vocab_size=10, n_ctx=8)
    >>> t["parameters_per_layer"], t["total_parameters"]
    (244, 568)
    >>> t["shape_trace"][-1]
    ('logits', (1, 10))

    References
    ----------
    Géron Ch 15
    """
    ids = np.atleast_1d(np.asarray(prompt)).ravel()
    if ids.size == 0:
        raise ValueError("geron_gpt3: prompt is empty; an autoregressive LM needs at least one token")
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.all(np.asarray(ids, dtype=float) == np.floor(np.asarray(ids, dtype=float))):
            raise ValueError("geron_gpt3: prompt must contain integer token ids")
        ids = ids.astype(np.int64)
    n_layers = int(n_layers)
    d_model = int(d_model)
    n_heads = int(n_heads)
    vocab_size = int(vocab_size)
    n_ctx = int(n_ctx)
    n_new = int(n_tokens)
    dtype_bytes = int(dtype_bytes)
    d_ff = 4 * d_model if d_ff is None else int(d_ff)
    for name, val in (
        ("n_layers", n_layers),
        ("d_model", d_model),
        ("n_heads", n_heads),
        ("vocab_size", vocab_size),
        ("n_ctx", n_ctx),
        ("d_ff", d_ff),
        ("dtype_bytes", dtype_bytes),
    ):
        if val < 1:
            raise ValueError(f"geron_gpt3: {name} must be a positive integer, got {val}")
    if n_new < 0:
        raise ValueError(f"geron_gpt3: n_tokens must be non-negative, got {n_new}")
    if d_model % n_heads:
        raise ValueError(
            f"geron_gpt3: d_model={d_model} is not divisible by n_heads={n_heads}, so the heads cannot be equal width"
        )
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise ValueError(
            f"geron_gpt3: token ids must lie in 0..{vocab_size - 1}, got range {int(ids.min())}..{int(ids.max())}"
        )
    n_prompt = int(ids.size)
    total_len = n_prompt + n_new
    if total_len > n_ctx:
        raise ValueError(
            f"geron_gpt3: prompt of {n_prompt} tokens plus {n_new} generated exceeds the {n_ctx}-token context window"
        )

    d_head = d_model // n_heads
    attn_w = 4 * d_model * d_model
    attn_b = 4 * d_model
    mlp_w = 2 * d_model * d_ff
    mlp_b = d_ff + d_model
    ln_per_layer = 2 * (2 * d_model)
    per_layer = attn_w + attn_b + mlp_w + mlp_b + ln_per_layer

    breakdown = {
        "token_embedding": vocab_size * d_model,
        "position_embedding": n_ctx * d_model,
        "attention_weights": n_layers * attn_w,
        "attention_biases": n_layers * attn_b,
        "feedforward_weights": n_layers * mlp_w,
        "feedforward_biases": n_layers * mlp_b,
        "layer_norms": n_layers * ln_per_layer + 2 * d_model,
        "output_head": 0,
    }
    total = int(sum(breakdown.values()))

    shape_trace = [
        ("token_ids", (n_prompt,)),
        ("embedded", (n_prompt, d_model)),
        ("q_per_head", (n_heads, n_prompt, d_head)),
        ("attention_scores", (n_heads, n_prompt, n_prompt)),
        ("attention_out", (n_prompt, d_model)),
        ("ffn_hidden", (n_prompt, d_ff)),
        ("block_out", (n_prompt, d_model)),
        ("logits", (max(n_new, 1) if n_new else n_prompt, vocab_size)),
    ]
    if n_new:
        shape_trace[-1] = ("logits", (n_new, vocab_size))

    kv_cache_bytes = 2 * n_layers * total_len * d_model * dtype_bytes
    flops_per_token = 2 * total

    return RichResult(
        title="GPT-3 architecture",
        summary_lines=[
            ("Total parameters", total),
            ("Layers x d_model", f"{n_layers} x {d_model}"),
            ("Head width", d_head),
            ("KV cache (bytes)", kv_cache_bytes),
        ],
        interpretation=(
            "Depth and width are fixed; only the sequence length changes with the prompt. "
            "The KV cache, not the weights, is what grows as generation proceeds."
        ),
        payload={
            "total_parameters": total,
            "parameters_per_layer": int(per_layer),
            "breakdown": breakdown,
            "shape_trace": shape_trace,
            "d_head": int(d_head),
            "n_prompt_tokens": n_prompt,
            "n_generated": n_new,
            "context_used": total_len,
            "kv_cache_bytes": int(kv_cache_bytes),
            "flops_per_token": int(flops_per_token),
            "estimate": float(total),
            "n": n_prompt,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmgpt3: GPT-3 decoder-only architecture -- exact parameter count, shape trace, KV-cache size"


# compact alias per ledger/NAMING.md
gerongpt3 = geron_gpt3
