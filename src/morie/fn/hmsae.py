# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stacked (deep) autoencoder with multiple encoding/decoding layers."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_stacked_autoencoder"]


def _lcg(shape, seed, scale=0.1):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def _train_layer(H, k, epochs, lr, seed):
    """One tied-weight autoencoder layer: h = tanh(HW + b), H_hat = h W^T + c."""
    n, d = H.shape
    W = _lcg((d, k), seed)
    b = np.zeros(k)
    c = np.zeros(d)
    losses = np.empty(epochs)
    for e in range(epochs):
        h = np.tanh(H @ W + b)
        rec = h @ W.T + c
        diff = rec - H
        losses[e] = float(np.mean(diff * diff))
        g = 2.0 * diff / (n * d)
        dW = g.T @ h  # decoder path
        dc = g.sum(axis=0)
        dz = (g @ W) * (1.0 - h * h)  # encoder path (tied weights share dW)
        dW = dW + H.T @ dz
        db = dz.sum(axis=0)
        W = W - lr * dW
        b = b - lr * db
        c = c - lr * dc
    return W, b, c, losses


def _forward(A, Ws, bs, cs):
    """Encode through every layer, then decode back through the tied mirror."""
    hs = [A]
    H = A
    for W, b in zip(Ws, bs):
        H = np.tanh(H @ W + b)
        hs.append(H)
    L = len(Ws)
    rs = [None] * (L + 1)
    rs[L] = H
    for i in range(L - 1, -1, -1):
        u = rs[i + 1] @ Ws[i].T + cs[i]
        rs[i] = u if i == 0 else np.tanh(u)
    return hs, rs


def _backward(A, Ws, bs, cs, hs, rs):
    """Exact gradients of the reconstruction MSE for the tied symmetric stack."""
    n, d = A.shape
    L = len(Ws)
    dW = [np.zeros_like(w) for w in Ws]
    db = [np.zeros_like(v) for v in bs]
    dc = [np.zeros_like(v) for v in cs]
    g = 2.0 * (rs[0] - A) / (n * d)
    for i in range(L):  # decoder half, outermost layer first
        du = g if i == 0 else g * (1.0 - rs[i] * rs[i])
        dc[i] += du.sum(axis=0)
        dW[i] += du.T @ rs[i + 1]
        g = du @ Ws[i]
    for i in range(L - 1, -1, -1):  # encoder half, innermost layer first
        dz = g * (1.0 - hs[i + 1] * hs[i + 1])
        dW[i] += hs[i].T @ dz
        db[i] += dz.sum(axis=0)
        g = dz @ Ws[i].T
    return dW, db, dc


def geron_stacked_autoencoder(X, hidden_sizes=(2,), epochs=200, lr=0.5, seed=0, finetune=True):
    """
    Stacked (deep) autoencoder with multiple encoding/decoding layers.

    Formula: symmetric encoder/decoder; greedy layer-wise pretraining

    Both halves of the recipe are executed. **Greedy layer-wise
    pretraining**: each layer is trained as its own tied-weight
    autoencoder (``h = tanh(HW + b)``, ``H_hat = h W^T + c``) on the codes
    produced by the layer below, so layer L never sees the raw input.
    **Fine-tuning**: the pretrained layers are unrolled into the
    symmetric encoder/decoder stack and trained end to end by
    backpropagation. Weight tying halves the parameter count and makes
    the decoder an exact mirror of the encoder by construction.

    Parameters
    ----------
    X : array-like
        Training data (n, d).
    hidden_sizes : sequence of int, default (2,)
        Encoder widths, outermost first; each >= 1 and no wider than the
        layer feeding it (an overcomplete layer can learn the identity).
    epochs : int, default 200
        Gradient steps per pretraining stage and for fine-tuning.
    lr : float, default 0.5
        Learning rate (> 0).
    seed : int, default 0
        LCG seed for the weights.
    finetune : bool, default True
        Run the end-to-end fine-tuning pass after pretraining.

    Returns
    -------
    result : RichResult
        Keys: codes, reconstruction, recon_error, layer_losses,
        finetune_losses, weights, estimate, n, method.

    Examples
    --------
    Points on a line in 2-D compress to one code unit and come back:

    >>> X = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]]
    >>> r = geron_stacked_autoencoder(X, hidden_sizes=(1,), epochs=400, lr=0.3)
    >>> r["codes"].shape
    (5, 1)
    >>> bool(r["recon_error"] < 0.02)
    True
    >>> bool(r["layer_losses"][0][-1] < r["layer_losses"][0][0])
    True
    >>> bool(r["finetune_losses"][-1] <= r["finetune_losses"][0])
    True

    Two stacked layers train without ever showing the raw input to the
    second one:

    >>> r2 = geron_stacked_autoencoder(X, hidden_sizes=(2, 1), epochs=200, lr=0.2)
    >>> r2["codes"].shape
    (5, 1)
    >>> len(r2["layer_losses"])
    2

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_stacked_autoencoder: X must be a non-empty (n, d) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_stacked_autoencoder: X contains non-finite values")
    sizes = [int(s) for s in np.atleast_1d(np.asarray(hidden_sizes)).ravel()]
    if not sizes:
        raise ValueError("geron_stacked_autoencoder: hidden_sizes is empty")
    widths = [A.shape[1]] + sizes
    for i in range(1, len(widths)):
        if widths[i] < 1:
            raise ValueError(f"geron_stacked_autoencoder: hidden size {widths[i]} must be >= 1")
        if widths[i] > widths[i - 1]:
            raise ValueError(
                f"geron_stacked_autoencoder: layer {i} widens {widths[i - 1]} -> {widths[i]}; "
                "an overcomplete layer can learn the identity and encode nothing"
            )
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_stacked_autoencoder: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_stacked_autoencoder: lr must be positive and finite, got {step}")

    Ws, bs, cs, layer_losses = [], [], [], []
    H = A
    for i, k in enumerate(sizes):
        W, b, c, losses = _train_layer(H, k, E, step, int(seed) + 17 * (i + 1))
        Ws.append(W)
        bs.append(b)
        cs.append(c)
        layer_losses.append(losses)
        H = np.tanh(H @ W + b)

    hs, rs = _forward(A, Ws, bs, cs)
    ft = [float(np.mean((rs[0] - A) ** 2))]
    if finetune:
        for _ in range(E):
            hs, rs = _forward(A, Ws, bs, cs)
            dW, db, dc = _backward(A, Ws, bs, cs, hs, rs)
            for i in range(len(Ws)):
                Ws[i] = Ws[i] - step * dW[i]
                bs[i] = bs[i] - step * db[i]
                cs[i] = cs[i] - step * dc[i]
            hs, rs = _forward(A, Ws, bs, cs)
            ft.append(float(np.mean((rs[0] - A) ** 2)))

    hs, rs = _forward(A, Ws, bs, cs)
    err = float(np.mean((rs[0] - A) ** 2))

    return RichResult(
        title="Stacked autoencoder",
        summary_lines=[
            ("Layers", len(sizes)),
            ("Widths", " -> ".join(str(w) for w in widths)),
            ("Reconstruction MSE", err),
        ],
        interpretation=(
            "Greedy pretraining gives each layer a sensible starting point before the deep stack is "
            "trained jointly; tied weights keep the decoder an exact mirror of the encoder."
        ),
        payload={
            "codes": hs[-1],
            "reconstruction": rs[0],
            "recon_error": err,
            "layer_losses": layer_losses,
            "finetune_losses": np.asarray(ft, dtype=float),
            "weights": Ws,
            "biases": bs,
            "decoder_biases": cs,
            "widths": widths,
            "estimate": err,
            "n": int(A.shape[0]),
            "method": "Tied-weight stacked autoencoder: greedy layer-wise pretraining then end-to-end fine-tuning",
        },
    )


def cheatsheet():
    return "hmsae: Stacked (deep) autoencoder with multiple encoding/decoding layers"
