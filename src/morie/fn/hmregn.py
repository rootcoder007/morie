# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression MLP with a linear output layer and MSE loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_regression_mlp"]


def _lcg_uniform(rows, cols, seed, scale):
    s = int(seed) % 2**32
    out = np.empty(rows * cols)
    for i in range(rows * cols):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = ((s + 0.5) / 2**32 * 2.0 - 1.0) * scale
    return out.reshape(rows, cols)


def geron_regression_mlp(X, y, hidden_sizes=(8,), epochs=400, lr=0.05, seed=0):
    """
    Regression MLP: linear output layer and MSE loss.

    Formula: loss = (1/m) sum ||y_hat - y||^2

    The output layer carries NO activation, which is the point of a
    regression head: a ReLU there would make negative predictions
    impossible and a sigmoid would bound them. The hidden layers use
    ReLU and He-scaled initialisation (uniform on +-sqrt(6/fan_in)),
    without which a deep stack's activations shrink layer by layer until
    the gradient reaching the first layer is numerically nothing.

    Training is full-batch gradient descent so the run is deterministic;
    the weights come from a reproducible integer LCG rather than a
    platform RNG.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,) or (m, k)
    hidden_sizes : sequence of int, default (8,)
        Hidden widths.
    epochs : int, default 400
    lr : float, default 0.05
        Learning rate (positive).
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: predict, predictions, mse, loss_history, weights, biases,
        n_parameters, estimate, n, method.

    Examples
    --------
    y = 2x is learned to a small error, and the loss falls monotonically
    on this convex-enough problem:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_regression_mlp(X, [2.0, 4.0, 6.0, 8.0], hidden_sizes=(8,), epochs=800, lr=0.02)
    >>> bool(r["mse"] < 0.05)
    True
    >>> bool(r["loss_history"][-1] < r["loss_history"][0])
    True

    The head is linear, so predictions are ordered like the input:

    >>> p = r["predict"]([[1.0], [4.0]])
    >>> bool(p[0] < p[1])
    True

    Parameter count for 1 -> 8 -> 1: (1*8+8) + (8*1+1) = 25

    >>> int(r["n_parameters"])
    25

    References
    ----------
    Geron Ch 9
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_regression_mlp: X must be a non-empty 2-D array, got shape {A.shape}")
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"geron_regression_mlp: y must be 1-D or 2-D, got ndim={Y.ndim}")
    if Y.shape[0] != A.shape[0]:
        raise ValueError(f"geron_regression_mlp: X has {A.shape[0]} rows but y has {Y.shape[0]}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Y)):
        raise ValueError("geron_regression_mlp: inputs contain non-finite values")
    hs = [int(h) for h in (hidden_sizes if np.ndim(hidden_sizes) else [hidden_sizes])]
    if any(h < 1 for h in hs):
        raise ValueError(f"geron_regression_mlp: hidden sizes must be >= 1, got {hidden_sizes!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_regression_mlp: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_regression_mlp: lr must be positive and finite, got {lr!r}")

    m = A.shape[0]
    sizes = [A.shape[1]] + hs + [Y.shape[1]]
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        scale = np.sqrt(6.0 / sizes[i])
        Ws.append(_lcg_uniform(sizes[i], sizes[i + 1], seed + 7919 * i + 1, scale))
        bs.append(np.zeros(sizes[i + 1]))

    def forward(B, _Ws=Ws, _bs=bs):
        acts = [B]
        z = B
        for i in range(len(_Ws)):
            z = acts[-1] @ _Ws[i] + _bs[i]
            acts.append(np.maximum(z, 0.0) if i < len(_Ws) - 1 else z)
        return acts

    hist = []
    for _ in range(E):
        acts = forward(A)
        pred = acts[-1]
        resid = pred - Y
        hist.append(float(np.mean(np.sum(resid**2, axis=1))))
        delta = (2.0 / m) * resid
        for i in range(len(Ws) - 1, -1, -1):
            gW = acts[i].T @ delta
            gb = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ Ws[i].T) * (acts[i] > 0)
            Ws[i] -= eta * gW
            bs[i] -= eta * gb

    acts = forward(A)
    pred = acts[-1]
    mse = float(np.mean(np.sum((pred - Y) ** 2, axis=1)))
    hist.append(mse)

    def predict(Xnew, _d=A.shape[1], _k=Y.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"predict: expected {_d} features, got {B.shape[1]}")
        out = forward(B)[-1]
        return out.ravel() if _k == 1 else out

    nparams = int(sum(W.size + b.size for W, b in zip(Ws, bs)))
    return RichResult(
        title="Regression MLP",
        summary_lines=[("Architecture", tuple(sizes)), ("Parameters", nparams), ("Training MSE", mse)],
        interpretation="No activation on the output layer; a ReLU there would forbid negative predictions.",
        payload={
            "predict": predict,
            "predictions": pred.ravel() if Y.shape[1] == 1 else pred,
            "mse": mse,
            "loss_history": hist,
            "weights": Ws,
            "biases": bs,
            "sizes": sizes,
            "n_parameters": nparams,
            "estimate": pred.ravel() if Y.shape[1] == 1 else pred,
            "n": int(m),
            "method": "Regression MLP (ReLU hidden, linear head) trained by full-batch gradient descent",
        },
    )


def cheatsheet():
    return "hmregn: Regression MLP with linear output and MSE loss"
