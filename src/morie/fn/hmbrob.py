# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RoBERTa: robustly-optimized BERT pretraining."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_roberta"]


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


def _block_weights(d_model, d_ff, seed):
    return {
        "Wq": _init((d_model, d_model), seed + 1),
        "Wk": _init((d_model, d_model), seed + 2),
        "Wv": _init((d_model, d_model), seed + 3),
        "Wo": _init((d_model, d_model), seed + 4),
        "W1": _init((d_model, d_ff), seed + 5),
        "b1": np.zeros(d_ff),
        "W2": _init((d_ff, d_model), seed + 6),
        "b2": np.zeros(d_model),
    }


def geron_roberta(X, n_layers=2, n_heads=2, d_model=8, vocab_size=None, d_ff=None, mask_prob=0.15, epochs=4, seed=0):
    """
    RoBERTa: robustly-optimized BERT pretraining.

    Formula: BERT without NSP, larger batches, longer training

    Same encoder stack as BERT with two deliberate differences, both visible
    in the returned payload: there is no next-sentence-prediction head (so
    the parameter count is smaller by d_model*2), and masking is *dynamic* --
    a fresh mask is drawn for every epoch instead of being frozen once during
    preprocessing, so the same sequence contributes a different MLM problem
    each pass.

    Parameters
    ----------
    X : array-like of int
        Token ids, shape (T,) or (batch, T).
    n_layers, n_heads, d_model, d_ff : int
        Encoder geometry; d_model must divide by n_heads.
    vocab_size : int, optional
        Defaults to max(X) + 1; one extra row is appended for [MASK].
    mask_prob : float
        Masking fraction in (0, 1).
    epochs : int
        Number of dynamic-masking passes (>= 1).
    seed : int
        LCG seed.

    Returns
    -------
    result : RichResult
        Keys: hidden, mlm_loss, epoch_losses, masks, dynamic_masking,
        has_nsp_head, n_params, estimate, n, method.

    Examples
    --------
    >>> ids = [1, 2, 3, 4, 5, 6, 0, 2]
    >>> r = geron_roberta(ids, epochs=3)
    >>> r["has_nsp_head"]
    False
    >>> len(r["masks"])
    3
    >>> bool(r["masks"][0] != r["masks"][2])
    True
    >>> r["hidden"].shape
    (1, 8, 8)
    >>> round(abs(float(r["hidden"][0, 3].mean())), 12)
    0.0

    Dropping the NSP head is exactly d_model*2 fewer parameters than the
    BERT-style count:

    >>> r["n_params_with_nsp"] - r["n_params"]
    16

    References
    ----------
    Géron Ch 15
    """
    ids = np.asarray(X)
    if ids.ndim == 1:
        ids = ids.reshape(1, -1)
    if ids.ndim != 2:
        raise ValueError(f"geron_roberta: X must be 1-D or 2-D token ids, got ndim={ids.ndim}")
    if ids.size == 0:
        raise ValueError("geron_roberta: X is empty")
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.all(np.asarray(ids, dtype=float) == np.floor(np.asarray(ids, dtype=float))):
            raise ValueError("geron_roberta: X must contain integer token ids")
        ids = ids.astype(int)
    if np.min(ids) < 0:
        raise ValueError("geron_roberta: token ids must be non-negative")
    B, T = ids.shape
    L, H, d = int(n_layers), int(n_heads), int(d_model)
    if L < 1 or H < 1 or d < 1:
        raise ValueError("geron_roberta: n_layers, n_heads and d_model must all be >= 1")
    if d % H:
        raise ValueError(f"geron_roberta: d_model={d} is not divisible by n_heads={H}")
    Vsz = int(np.max(ids)) + 1 if vocab_size is None else int(vocab_size)
    if Vsz <= int(np.max(ids)):
        raise ValueError(f"geron_roberta: vocab_size={Vsz} is too small for token id {int(np.max(ids))}")
    ff = 2 * d if d_ff is None else int(d_ff)
    if ff < 1:
        raise ValueError("geron_roberta: d_ff must be >= 1")
    p = float(mask_prob)
    if not (0.0 < p < 1.0):
        raise ValueError(f"geron_roberta: mask_prob must lie in (0, 1), got {p}")
    EP = int(epochs)
    if EP < 1:
        raise ValueError("geron_roberta: epochs must be >= 1")
    n_mask = max(1, int(round(p * T)))
    if n_mask >= T and T > 1:
        raise ValueError("geron_roberta: mask_prob would mask the entire sequence")

    mask_id = Vsz
    E = _init((Vsz + 1, d), seed + 101)
    P = _init((T, d), seed + 202)
    blocks = [_block_weights(d, ff, seed + 1000 * (i + 1)) for i in range(L)]

    hidden = np.empty((B, T, d))
    masks = []
    epoch_losses = np.empty(EP)
    for ep in range(EP):
        losses = []
        for b in range(B):
            # Fresh draw per (epoch, sequence): this is the dynamic masking.
            u = _lcg(T, seed + 977 * (ep + 1) + 31 * b, scale=0.5) + 0.5
            pos = np.sort(np.argsort(u, kind="mergesort")[:n_mask])
            if b == 0:
                masks.append(pos.tolist())
            toks = ids[b].copy()
            toks[pos] = mask_id
            Xh = E[toks] + P
            for W in blocks:
                Xh = _encoder_block(Xh, W, H)
            if ep == EP - 1:
                hidden[b] = Xh
            probs = _softmax(Xh[pos] @ E.T, axis=-1)
            losses.append(float(-np.mean(np.log(np.clip(probs[np.arange(len(pos)), ids[b][pos]], 1e-15, None)))))
        epoch_losses[ep] = float(np.mean(losses))

    base = int((Vsz + 1) * d + T * d + L * (4 * d * d + d * ff + ff + ff * d + d))

    return RichResult(
        title="RoBERTa pretraining",
        summary_lines=[("Layers", L), ("Epochs", EP), ("Dynamic masking", True), ("MLM loss", float(epoch_losses[-1]))],
        payload={
            "hidden": hidden,
            "mlm_loss": float(epoch_losses[-1]),
            "epoch_losses": epoch_losses,
            "masks": masks,
            "dynamic_masking": True,
            "has_nsp_head": False,
            "n_params": base,
            "n_params_with_nsp": base + d * 2,
            "n_masked_per_sequence": n_mask,
            "estimate": float(epoch_losses[-1]),
            "n": int(B * T),
            "method": "RoBERTa: NSP-free encoder pretraining with per-epoch dynamic masking",
        },
    )


def cheatsheet():
    return "hmbrob: RoBERTa: robustly-optimized BERT pretraining"
