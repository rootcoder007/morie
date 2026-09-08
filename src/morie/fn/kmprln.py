# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-LayerNorm transformer block (the stable-training placement)."""

from . import _array_core as np

from ._richresult import RichResult
from .kmpoln import layer_norm

__all__ = ["kamath_pre_ln_transformer"]


def kamath_pre_ln_transformer(x, attn_fn, ffn_fn, eps=1e-5):
    """y = x + Attn(LN(x));  z = y + FFN(LN(y)).

    Pre-LN normalises the sublayer INPUT and leaves the residual
    stream untouched, so a signal can pass from the first block to the
    last without being rescaled -- that identity path is why pre-LN
    trains without a warmup. The LN itself is shared with
    ``morie.fn.kmpoln`` rather than re-derived, so the only difference
    between the two blocks is the one that matters: placement.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, pre-LN vs post-LN.

    Examples
    --------
    >>> ident = lambda a: a
    >>> zero = lambda a: [[0.0, 0.0]]
    >>> out = kamath_pre_ln_transformer([[3.0, 1.0]], zero, zero, eps=0.0)
    >>> out["output"]
    [[3.0, 1.0]]
    >>> out2 = kamath_pre_ln_transformer([[3.0, 1.0]], ident, zero, eps=0.0)
    >>> out2["output"]
    [[4.0, 0.0]]
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

    y = x + _sub(attn_fn, layer_norm(x, eps), "attn_fn")
    z = y + _sub(ffn_fn, layer_norm(y, eps), "ffn_fn")
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in z],
        "after_attention": [[float(v) for v in row] for row in y],
        "estimate": float(z[0, 0]),
        "placement": "pre-LN", "eps": float(eps),
        "n": int(z.shape[0]),
        "method": "Pre-LayerNorm transformer block"})


def cheatsheet():
    return "kmprln: z = y + FFN(LN(y)), y = x + Attn(LN(x)); clean residual"
