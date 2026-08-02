# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transformer decoder block: masked self-attention, cross-attention, FFN."""

from . import _array_core as np

from ._richresult import RichResult
from .grteb import feed_forward, layer_norm, multi_head_attention

__all__ = ["geron_transformer_decoder_block"]

_METHOD = "Transformer decoder block (post-norm)"


def geron_transformer_decoder_block(x, encoder_output, weights, eps=1e-5):
    r"""One decoder layer: three sublayers, each residual + LayerNorm.

    .. math::
        h_1 &= \mathrm{LayerNorm}(x + \mathrm{MaskedMHA}(x))\\
        h_2 &= \mathrm{LayerNorm}(h_1 + \mathrm{CrossAttn}(h_1, \text{enc}))\\
        y   &= \mathrm{LayerNorm}(h_2 + \mathrm{FFN}(h_2))

    The causal mask is generated here rather than asked for, because a
    decoder that can see the future trains to a beautiful loss and
    generates nothing: position ``t`` may attend to ``<= t`` only.  The
    cross-attention takes queries from the decoder and keys/values from
    the encoder, and it is *not* masked -- the whole source sequence is
    available at every output step.  Sublayer machinery is shared with
    :mod:`morie.fn.grteb`.

    Parameters
    ----------
    x : array-like, shape (T, d_model)
    encoder_output : array-like, shape (S, d_model)
    weights : dict
        ``self`` and ``cross`` attention weight dicts (``WQ``/``WK``/
        ``WV`` head lists plus ``WO``) and ``ffn``.
    eps : float, optional

    Returns
    -------
    RichResult
        Payload keys ``output``, ``self_attention_weights``,
        ``cross_attention_weights``, ``h1``, ``h2``, ``causal_mask``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Encoder-decoder transformer section.

    Examples
    --------
    Identity projections, zero FFN, 2 decoder tokens:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> att = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    >>> W = {"self": att, "cross": att, "ffn": {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]}}
    >>> r = geron_transformer_decoder_block(I, I, W)
    >>> r["causal_mask"]
    [[True, False], [True, True]]

    Token 0 attends only to itself:

    >>> r["self_attention_weights"][0][0]
    [1.0, 0.0]
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    E = np.atleast_2d(np.asarray(encoder_output, dtype=float))
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"x must be a non-empty (T, d_model) matrix, got shape {X.shape}.")
    if E.ndim != 2 or E.size == 0:
        raise ValueError(f"encoder_output must be a non-empty (S, d_model) matrix, got {E.shape}.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(E)):
        raise ValueError("x and encoder_output must be finite.")
    if not isinstance(weights, dict):
        raise ValueError(f"weights must be a dict, got {type(weights).__name__}.")
    missing = {"self", "cross", "ffn"} - set(weights)
    if missing:
        raise ValueError(f"weights missing keys: {sorted(missing)}.")

    T = X.shape[0]
    causal = np.tril(np.ones((T, T), dtype=bool))
    sa, sw = multi_head_attention(X, X, weights["self"], causal)
    if sa.shape != X.shape:
        raise ValueError(f"masked self-attention returned {sa.shape}, residual needs {X.shape}.")
    h1 = layer_norm(X + sa, weights["self"].get("gamma"), weights["self"].get("beta"), eps)

    ca, cw = multi_head_attention(h1, E, weights["cross"], None)
    if ca.shape != X.shape:
        raise ValueError(f"cross-attention returned {ca.shape}, residual needs {X.shape}.")
    h2 = layer_norm(h1 + ca, weights["cross"].get("gamma"), weights["cross"].get("beta"), eps)

    f = feed_forward(h2, weights["ffn"])
    y = layer_norm(h2 + f, weights["ffn"].get("gamma"), weights["ffn"].get("beta"), eps)

    return RichResult(
        title="Transformer decoder block",
        summary_lines=[("Target tokens", int(T)), ("Source tokens", int(E.shape[0]))],
        payload={
            "output": y.tolist(),
            "self_attention_weights": [w.tolist() for w in sw],
            "cross_attention_weights": [w.tolist() for w in cw],
            "h1": h1.tolist(),
            "h2": h2.tolist(),
            "causal_mask": causal.tolist(),
            "estimate": y.tolist(),
            "n": int(T),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtdb: LN(x+maskedMHA) -> LN(h1+cross(h1,enc)) -> LN(h2+FFN); causal mask built in, cross unmasked"
