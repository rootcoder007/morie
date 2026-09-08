# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-LayerNorm transformer block (the original Vaswani
placement)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_post_ln_transformer", "layer_norm"]


def layer_norm(x, eps=1e-5):
    """Standardise each row over the feature axis."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def kamath_post_ln_transformer(x, attn_fn, ffn_fn, eps=1e-5):
    """y = LN(x + Attn(x));  z = LN(y + FFN(y)).

    Post-LN normalises AFTER the residual add, so the residual stream
    is rescaled at every block -- the placement that makes deep
    post-LN stacks need a warmup schedule. ``morie.fn.kmprln`` is the
    pre-LN sibling; the two differ only in where LN sits, and running
    both on the same input is the cleanest way to see it.

    ``attn_fn`` and ``ffn_fn`` are the caller's sublayers, each
    ``array -> array`` of the same shape; a sublayer that changes the
    shape breaks the residual connection and is refused.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, pre-LN vs post-LN.

    Examples
    --------
    >>> out = kamath_post_ln_transformer([[1.0, -1.0]],
    ...     lambda a: [[0.0, 0.0]], lambda a: [[0.0, 0.0]], eps=0.0)
    >>> out["output"]
    [[1.0, -1.0]]
    >>> out2 = kamath_post_ln_transformer([[3.0, 1.0]],
    ...     lambda a: [[0.0, 0.0]], lambda a: [[0.0, 0.0]], eps=0.0)
    >>> out2["output"]
    [[1.0, -1.0]]
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if not callable(attn_fn) or not callable(ffn_fn):
        raise ValueError("attn_fn and ffn_fn must be callables.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    def _sub(f, v, name):
        out = np.atleast_2d(np.asarray(f(v), dtype=float))
        if out.shape != v.shape:
            raise ValueError(
                f"{name} returned {out.shape} for a {v.shape} input; the "
                "residual connection needs the shape preserved.")
        return out

    a = _sub(attn_fn, x, "attn_fn")
    y = layer_norm(x + a, eps)
    f = _sub(ffn_fn, y, "ffn_fn")
    z = layer_norm(y + f, eps)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in z],
        "after_attention": [[float(v) for v in row] for row in y],
        "estimate": float(z[0, 0]),
        "placement": "post-LN", "eps": float(eps),
        "n": int(z.shape[0]),
        "method": "Post-LayerNorm transformer block"})


def cheatsheet():
    return "kmpoln: z = LN(LN(x+Attn(x)) + FFN(.)); LN after the add"
