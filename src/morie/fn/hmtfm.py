# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transformer architecture (Vaswani et al. 2017)."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsdp import geron_scaled_dot_product

__all__ = ["geron_transformer", "encoder_params"]


def _lcg(shape, seed, scale=0.1):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def _layernorm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    return (x - mu) / sd


def encoder_params(d_model, d_ff, n_layers):
    """Parameter count of an encoder stack (weights + biases, no embeddings).

    Per layer: 4 projections of ``d_model x d_model`` (Q, K, V, O) plus a
    two-layer feedforward ``d_model -> d_ff -> d_model`` with biases, plus
    two layer-norm scale/shift pairs of length ``d_model``.
    """
    per = 4 * d_model * d_model + d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model
    return int(per * n_layers)


def geron_transformer(X, n_heads=2, d_model=None, n_layers=2, d_ff=None, seed=0, mask=None):
    """
    Transformer architecture (Vaswani et al. 2017).

    Formula: stacked multi-head attention + feedforward with residual connections

    Runs the real encoder stack -- multi-head attention (heads delegated
    to :func:`morie.fn.hmsdp.geron_scaled_dot_product`), residual add,
    layer norm, position-wise feedforward, residual add, layer norm --
    on deterministic LCG weights, and resolves the architecture to an
    exact parameter count via :func:`encoder_params`.

    Parameters
    ----------
    X : array-like
        Input sequence of embeddings, shape (T, d_model).
    n_heads : int, default 2
        Attention heads; must divide ``d_model``.
    d_model : int, optional
        Model width; defaults to (and must match) ``X.shape[1]``.
    n_layers : int, default 2
        Encoder blocks (>= 1).
    d_ff : int, optional
        Feedforward width; defaults to ``4 * d_model``.
    seed : int, default 0
        LCG seed for the weights.
    mask : array-like, optional
        (T, T) visibility mask forwarded to every head.

    Returns
    -------
    result : RichResult
        Keys: Y, attention, total_params, d_ff, estimate, n, method.

    Examples
    --------
    >>> X = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    >>> r = geron_transformer(X, n_heads=2, n_layers=1)
    >>> r["Y"].shape
    (3, 4)
    >>> int(r["total_params"])
    228
    >>> [round(float(v), 12) for v in r["attention"][0][0].sum(axis=1)]
    [1.0, 1.0, 1.0]

    Layer norm leaves each output row with zero mean:

    >>> bool(abs(r["Y"].mean(axis=1)).max() < 1e-12)
    True

    References
    ----------
    Géron Ch 15
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(1, -1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_transformer: X must be a non-empty (T, d_model) matrix")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_transformer: X contains non-finite values")
    d = int(Xa.shape[1]) if d_model is None else int(d_model)
    if d != Xa.shape[1]:
        raise ValueError(f"geron_transformer: d_model={d} but X has width {Xa.shape[1]}")
    h = int(n_heads)
    if h < 1:
        raise ValueError(f"geron_transformer: n_heads must be >= 1, got {h}")
    if d % h:
        raise ValueError(f"geron_transformer: n_heads={h} does not divide d_model={d}")
    L = int(n_layers)
    if L < 1:
        raise ValueError(f"geron_transformer: n_layers must be >= 1, got {L}")
    ff = 4 * d if d_ff is None else int(d_ff)
    if ff < 1:
        raise ValueError(f"geron_transformer: d_ff must be >= 1, got {ff}")

    dh = d // h
    H = Xa.copy()
    attn_all = []
    for layer in range(L):
        base = int(seed) + 1000 * layer
        Wq, Wk, Wv, Wo = (_lcg((d, d), base + i) for i in range(1, 5))
        W1 = _lcg((d, ff), base + 5)
        W2 = _lcg((ff, d), base + 6)
        Q, K, V = H @ Wq, H @ Wk, H @ Wv
        out = np.empty_like(Q)
        heads = np.empty((h, H.shape[0], H.shape[0]))
        for j in range(h):
            sl = slice(j * dh, (j + 1) * dh)
            a = geron_scaled_dot_product(Q[:, sl], K[:, sl], V[:, sl], d_k=dh, mask=mask)
            out[:, sl] = np.asarray(a["Y"], dtype=float)
            heads[j] = np.asarray(a["attention"], dtype=float)
        attn_all.append(heads)
        H = _layernorm(H + out @ Wo)
        H = _layernorm(H + np.maximum(H @ W1, 0.0) @ W2)

    total = encoder_params(d, ff, L)

    return RichResult(
        title="Transformer encoder stack",
        summary_lines=[
            ("Tokens", int(Xa.shape[0])),
            ("d_model", d),
            ("Heads", h),
            ("Layers", L),
            ("d_ff", ff),
            ("Parameters", total),
        ],
        interpretation=(
            "Residual connections plus layer norm keep the signal scale stable through the stack; "
            "the parameter count is dominated by 4*d_model^2 attention weights and 2*d_model*d_ff feedforward weights."
        ),
        payload={
            "Y": H,
            "attention": attn_all,
            "total_params": total,
            "d_model": d,
            "d_ff": ff,
            "n_heads": h,
            "n_layers": L,
            "estimate": float(total),
            "n": int(Xa.shape[0]),
            "method": "Post-norm transformer encoder stack with exact parameter accounting",
        },
    )


def cheatsheet():
    return "hmtfm: Transformer architecture (Vaswani et al. 2017)"


# compact alias per ledger/NAMING.md
encoderparams = encoder_params
