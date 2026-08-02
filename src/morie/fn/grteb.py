# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transformer encoder block: multi-head attention + FFN, both residual + LayerNorm."""

from . import _array_core as np

from ._richresult import RichResult
from .grsdpa import attend

__all__ = ["geron_transformer_encoder_block", "layer_norm", "multi_head_attention", "feed_forward"]

_METHOD = "Transformer encoder block (post-norm)"


def layer_norm(X, gamma=None, beta=None, eps=1e-5):
    """Normalise each row to zero mean, unit variance, then scale and shift.

    Per *token*, not per batch -- which is why it works with sequence
    length 1 and why transformers use it instead of batch norm.
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"layer_norm needs a non-empty 2-D array, got shape {A.shape}.")
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    mu = A.mean(axis=1, keepdims=True)
    var = A.var(axis=1, keepdims=True)
    Z = (A - mu) / np.sqrt(var + eps)
    d = A.shape[1]
    if gamma is not None:
        g = np.asarray(gamma, dtype=float).ravel()
        if g.size != d:
            raise ValueError(f"gamma must have {d} entries, got {g.size}.")
        Z = Z * g
    if beta is not None:
        b = np.asarray(beta, dtype=float).ravel()
        if b.size != d:
            raise ValueError(f"beta must have {d} entries, got {b.size}.")
        Z = Z + b
    return Z


def multi_head_attention(Q_in, KV_in, weights, mask=None):
    """Run ``h`` attention heads and mix them with ``WO``.

    ``weights`` is a dict with ``WQ``, ``WK``, ``WV`` (each a list of
    per-head projection matrices) and ``WO``. Heads are concatenated in
    order before ``WO``, which is the convention that makes the
    single-head case identical to :mod:`morie.fn.grsdpa`.
    """
    if not isinstance(weights, dict):
        raise ValueError(f"attention weights must be a dict, got {type(weights).__name__}.")
    missing = {"WQ", "WK", "WV", "WO"} - set(weights)
    if missing:
        raise ValueError(f"attention weights missing keys: {sorted(missing)}.")
    Qi = np.atleast_2d(np.asarray(Q_in, dtype=float))
    Ki = np.atleast_2d(np.asarray(KV_in, dtype=float))
    heads_q, heads_k, heads_v = (list(weights[k]) for k in ("WQ", "WK", "WV"))
    if not (len(heads_q) == len(heads_k) == len(heads_v)):
        raise ValueError(
            f"WQ/WK/WV must list the same number of heads, got "
            f"{len(heads_q)}, {len(heads_k)}, {len(heads_v)}."
        )
    if not heads_q:
        raise ValueError("no attention heads supplied.")
    outs, ws = [], []
    for hq, hk, hv in zip(heads_q, heads_k, heads_v):
        A = np.atleast_2d(np.asarray(hq, dtype=float))
        B = np.atleast_2d(np.asarray(hk, dtype=float))
        C = np.atleast_2d(np.asarray(hv, dtype=float))
        if A.shape[0] != Qi.shape[1]:
            raise ValueError(f"WQ head has {A.shape[0]} rows but queries have width {Qi.shape[1]}.")
        if B.shape[0] != Ki.shape[1] or C.shape[0] != Ki.shape[1]:
            raise ValueError(f"WK/WV heads must have {Ki.shape[1]} rows to match the key input.")
        o, w = attend(Qi @ A, Ki @ B, Ki @ C, mask)
        outs.append(o)
        ws.append(w)
    concat = np.hstack(outs)
    WO = np.atleast_2d(np.asarray(weights["WO"], dtype=float))
    if WO.shape[0] != concat.shape[1]:
        raise ValueError(
            f"WO must have {concat.shape[1]} rows (concatenated head width), got {WO.shape[0]}."
        )
    return concat @ WO, ws


def feed_forward(X, weights):
    """Position-wise ``max(0, X W1 + b1) W2 + b2``."""
    if not isinstance(weights, dict):
        raise ValueError(f"ffn weights must be a dict, got {type(weights).__name__}.")
    missing = {"W1", "W2"} - set(weights)
    if missing:
        raise ValueError(f"ffn weights missing keys: {sorted(missing)}.")
    A = np.atleast_2d(np.asarray(X, dtype=float))
    W1 = np.atleast_2d(np.asarray(weights["W1"], dtype=float))
    W2 = np.atleast_2d(np.asarray(weights["W2"], dtype=float))
    if W1.shape[0] != A.shape[1]:
        raise ValueError(f"W1 must have {A.shape[1]} rows, got {W1.shape[0]}.")
    if W2.shape[0] != W1.shape[1]:
        raise ValueError(f"W2 must have {W1.shape[1]} rows to match the hidden width.")
    if W2.shape[1] != A.shape[1]:
        raise ValueError(
            f"W2 must map back to d_model={A.shape[1]} for the residual to add, got {W2.shape[1]}."
        )
    b1 = np.asarray(weights.get("b1", np.zeros(W1.shape[1])), dtype=float).ravel()
    b2 = np.asarray(weights.get("b2", np.zeros(W2.shape[1])), dtype=float).ravel()
    if b1.size != W1.shape[1] or b2.size != W2.shape[1]:
        raise ValueError("ffn biases do not match their weight widths.")
    return np.maximum(0.0, A @ W1 + b1) @ W2 + b2


def geron_transformer_encoder_block(x, mha_weights, ffn_weights, mask=None, eps=1e-5):
    r"""One post-norm encoder layer.

    .. math::
        h &= \mathrm{LayerNorm}(x + \mathrm{MHA}(x))\\
        y &= \mathrm{LayerNorm}(h + \mathrm{FFN}(h))

    Two sublayers, each wrapped in "add then normalise".  The residual is
    what lets the block start near the identity; the LayerNorm -- per
    token, so it does not care about batch size or sequence length -- is
    what stops the repeated additions from drifting in scale.  The FFN is
    the only place where positions are processed independently, which is
    why it can be applied as a plain matrix multiply.

    Parameters
    ----------
    x : array-like, shape (T, d_model)
    mha_weights : dict
        ``WQ``/``WK``/``WV`` lists of per-head matrices, plus ``WO``, and
        optionally ``gamma``/``beta`` for the first LayerNorm.
    ffn_weights : dict
        ``W1``, ``W2``, optional ``b1``, ``b2``, ``gamma``, ``beta``.
    mask : array-like of bool, optional
    eps : float, optional

    Returns
    -------
    RichResult
        Payload keys ``output``, ``attention_output``,
        ``attention_weights``, ``hidden``, ``ffn_output``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Encoder-only transformer section.

    Examples
    --------
    Identity projections and a zero FFN leave a LayerNorm'd residual:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> mha = {"WQ": [I], "WK": [I], "WV": [I], "WO": I}
    >>> ffn = {"W1": [[0.0], [0.0]], "W2": [[0.0, 0.0]]}
    >>> r = geron_transformer_encoder_block(I, mha, ffn)
    >>> [round(v, 6) for v in r["output"][0]]
    [0.999995, -0.999995]

    Every output row is zero-mean by construction (that is the LayerNorm):

    >>> [round(sum(row), 12) for row in r["output"]]
    [0.0, 0.0]
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"x must be a non-empty (T, d_model) matrix, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains non-finite values.")

    attn, ws = multi_head_attention(X, X, mha_weights, mask)
    if attn.shape != X.shape:
        raise ValueError(
            f"attention output has shape {attn.shape} but the residual needs {X.shape}; "
            "check WO."
        )
    h = layer_norm(X + attn, mha_weights.get("gamma"), mha_weights.get("beta"), eps)
    f = feed_forward(h, ffn_weights)
    y = layer_norm(h + f, ffn_weights.get("gamma"), ffn_weights.get("beta"), eps)

    return RichResult(
        title="Transformer encoder block",
        summary_lines=[("Tokens", int(X.shape[0])), ("Heads", len(ws))],
        payload={
            "output": y.tolist(),
            "attention_output": attn.tolist(),
            "attention_weights": [w.tolist() for w in ws],
            "hidden": h.tolist(),
            "ffn_output": f.tolist(),
            "estimate": y.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grteb: h=LN(x+MHA(x)); y=LN(h+FFN(h)); LayerNorm is per token, FFN is per position"
