# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ALBERT: cross-layer parameter sharing + factorized embeddings."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_albert"]


def _lcg(size, seed, scale=0.1):
    s = int(seed) % 2**32
    out = np.empty(int(size))
    for i in range(int(size)):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out


def _init(shape, seed, scale=0.1):
    return _lcg(int(np.prod(shape)), seed, scale).reshape(shape)


def _softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    sd = np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    return (x - mu) / sd


def _encoder_block(X, W, n_heads):
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ W["Wq"], X @ W["Wk"], X @ W["Wv"]
    out = np.empty_like(Q)
    for h in range(n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        a = _softmax(Q[:, sl] @ K[:, sl].T / np.sqrt(dh), axis=-1)
        out[:, sl] = a @ V[:, sl]
    X = _layernorm(X + out @ W["Wo"])
    f = np.maximum(X @ W["W1"] + W["b1"], 0.0) @ W["W2"] + W["b2"]
    return _layernorm(X + f)


def geron_albert(X, n_layers=4, n_heads=2, d_model=8, d_embed=4, vocab_size=None, d_ff=None, seed=0):
    """
    ALBERT: cross-layer parameter sharing + factorized embeddings.

    Formula: embed: V -> E (small) -> H; share weights across layers

    Both ALBERT tricks are real here, not annotated: the embedding is
    factorised through a low-rank bottleneck (V x E followed by E x H instead
    of V x H), and *one* transformer block object is applied `n_layers`
    times, so depth costs no extra parameters at all.

    Parameters
    ----------
    X : array-like of int
        Token ids, shape (T,) or (batch, T).
    n_layers : int
        Number of applications of the shared block (>= 1).
    n_heads, d_model, d_ff : int
        Attention geometry; d_model must divide by n_heads.
    d_embed : int
        Factorised embedding width E; must satisfy 1 <= E <= d_model
        (E > H would defeat the factorisation).
    vocab_size : int, optional
        Defaults to max(X) + 1.
    seed : int
        LCG seed.

    Returns
    -------
    result : RichResult
        Keys: hidden, n_params, n_params_unshared, embedding_params,
        embedding_params_direct, shared, estimate, n, method.

    Examples
    --------
    >>> ids = [0, 1, 2, 3]
    >>> r = geron_albert(ids, n_layers=4, n_heads=2, d_model=8, d_embed=4, vocab_size=5)
    >>> r["shared"]
    True
    >>> r["hidden"].shape
    (1, 4, 8)

    Depth is free: 4 shared layers cost the same as 1, while an unshared
    stack costs 4x the block:

    >>> r1 = geron_albert(ids, n_layers=1, n_heads=2, d_model=8, d_embed=4, vocab_size=5)
    >>> r["n_params"] == r1["n_params"]
    True
    >>> r["n_params_unshared"] - r["n_params"] == 3 * r["block_params"]
    True

    Factorising V -> E -> H is cheaper than V -> H once V is large:

    >>> big = geron_albert([0, 1, 2, 3], vocab_size=30000, d_model=8, d_embed=4)
    >>> bool(big["embedding_params"] < big["embedding_params_direct"])
    True

    References
    ----------
    Géron Ch 15
    """
    ids = np.asarray(X)
    if ids.ndim == 1:
        ids = ids.reshape(1, -1)
    if ids.ndim != 2:
        raise ValueError(f"geron_albert: X must be 1-D or 2-D token ids, got ndim={ids.ndim}")
    if ids.size == 0:
        raise ValueError("geron_albert: X is empty")
    if not np.issubdtype(ids.dtype, np.integer):
        ids = np.asarray(ids, dtype=float)
        if not np.all(ids == np.floor(ids)):
            raise ValueError("geron_albert: X must contain integer token ids")
        ids = ids.astype(int)
    if np.min(ids) < 0:
        raise ValueError("geron_albert: token ids must be non-negative")
    B, T = ids.shape
    L, H, d = int(n_layers), int(n_heads), int(d_model)
    if L < 1 or H < 1 or d < 1:
        raise ValueError("geron_albert: n_layers, n_heads and d_model must all be >= 1")
    if d % H:
        raise ValueError(f"geron_albert: d_model={d} is not divisible by n_heads={H}")
    E = int(d_embed)
    if E < 1 or E > d:
        raise ValueError(f"geron_albert: d_embed must lie in [1, d_model={d}], got {E}")
    Vsz = int(np.max(ids)) + 1 if vocab_size is None else int(vocab_size)
    if Vsz <= int(np.max(ids)):
        raise ValueError(f"geron_albert: vocab_size={Vsz} is too small for token id {int(np.max(ids))}")
    ff = 2 * d if d_ff is None else int(d_ff)
    if ff < 1:
        raise ValueError("geron_albert: d_ff must be >= 1")

    Wemb = _init((Vsz, E), seed + 101)
    Wproj = _init((E, d), seed + 202)
    Wpos = _init((T, d), seed + 303)
    shared = {
        "Wq": _init((d, d), seed + 1),
        "Wk": _init((d, d), seed + 2),
        "Wv": _init((d, d), seed + 3),
        "Wo": _init((d, d), seed + 4),
        "W1": _init((d, ff), seed + 5),
        "b1": np.zeros(ff),
        "W2": _init((ff, d), seed + 6),
        "b2": np.zeros(d),
    }

    hidden = np.empty((B, T, d))
    for b in range(B):
        Xh = (Wemb[ids[b]] @ Wproj) + Wpos
        for _ in range(L):
            Xh = _encoder_block(Xh, shared, H)
        hidden[b] = Xh

    block_params = int(4 * d * d + d * ff + ff + ff * d + d)
    emb_params = int(Vsz * E + E * d)
    emb_direct = int(Vsz * d)
    total = int(emb_params + T * d + block_params)

    return RichResult(
        title="ALBERT encoder",
        summary_lines=[
            ("Layers (shared)", L),
            ("Parameters", total),
            ("Unshared equivalent", emb_params + T * d + L * block_params),
        ],
        payload={
            "hidden": hidden,
            "n_params": total,
            "n_params_unshared": int(emb_params + T * d + L * block_params),
            "block_params": block_params,
            "embedding_params": emb_params,
            "embedding_params_direct": emb_direct,
            "shared": True,
            "d_embed": E,
            "estimate": float(total),
            "n": int(B * T),
            "method": "ALBERT: factorised embedding plus one cross-layer-shared encoder block",
        },
    )


def cheatsheet():
    return "hmalbt: ALBERT: cross-layer parameter sharing + factorized embeddings"
