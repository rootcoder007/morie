# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transfer learning: reuse pretrained model, fine-tune on new task."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_transfer_learning"]


def geron_transfer_learning(pretrained_model, X, y, n_frozen=1, epochs=200, lr=0.05):
    """
    Transfer learning: reuse pretrained model, fine-tune on new task.

    Formula: freeze initial layers; train final layers on new data

    Freezing is enforced, not documented: the first `n_frozen` weight
    matrices are used in the forward pass and never receive an update, so
    the returned weights for those layers are bit-identical to the ones
    passed in. Only the trainable tail is optimised, by backpropagation
    through the tanh stack (the frozen layers still propagate the signal
    forward, they just do not accept gradient).

    `pretrained_model` is the list of weight matrices, input side first;
    consecutive shapes must chain. The number of trainable parameters is
    reported, because that -- not the total -- is what determines how much
    data the fine-tuning needs.

    Parameters
    ----------
    pretrained_model : sequence of array-like
        Weight matrices ``W_0 (d_in, h_1), W_1 (h_1, h_2), ...``.
    X : array-like
        New-task inputs (n, d_in).
    y : array-like
        New-task targets (n,) or (n, d_out).
    n_frozen : int, default 1
        Leading layers to freeze (0 <= n_frozen < n_layers).
    epochs : int, default 200
        Gradient steps (>= 1).
    lr : float, default 0.05
        Learning rate (> 0).

    Returns
    -------
    result : RichResult
        Keys: weights, frozen, loss_curve, initial_loss, final_loss,
        trainable_params, total_params, estimate, n, method.

    Examples
    --------
    Two layers, the first frozen: layer 0 comes back untouched, layer 1
    has moved, and the loss has fallen.

    >>> import numpy as np
    >>> W0 = [[0.5, -0.5], [0.5, 0.5]]
    >>> W1 = [[1.0], [1.0]]
    >>> X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    >>> y = [1.0, 2.0, 3.0, 4.0]
    >>> r = geron_transfer_learning([W0, W1], X, y, n_frozen=1, epochs=300, lr=0.1)
    >>> bool(np.array_equal(r["weights"][0], np.asarray(W0)))
    True
    >>> bool(not np.array_equal(r["weights"][1], np.asarray(W1)))
    True
    >>> bool(r["final_loss"] < r["initial_loss"])
    True
    >>> int(r["trainable_params"]), int(r["total_params"])
    (2, 6)

    References
    ----------
    Géron Ch 11
    """
    Ws = [np.asarray(w, dtype=float) for w in pretrained_model]
    if len(Ws) < 2:
        raise ValueError("geron_transfer_learning: pretrained_model needs at least 2 layers to freeze part of it")
    for i, w in enumerate(Ws):
        if w.ndim != 2:
            raise ValueError(f"geron_transfer_learning: layer {i} must be a 2-D weight matrix, got shape {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"geron_transfer_learning: layer {i} contains non-finite weights")
    for i in range(1, len(Ws)):
        if Ws[i - 1].shape[1] != Ws[i].shape[0]:
            raise ValueError(
                f"geron_transfer_learning: layer {i - 1} outputs {Ws[i - 1].shape[1]} units but layer {i} "
                f"expects {Ws[i].shape[0]}"
            )
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_transfer_learning: X must be a non-empty (n, d_in) matrix")
    if A.shape[1] != Ws[0].shape[0]:
        raise ValueError(f"geron_transfer_learning: X has {A.shape[1]} features but layer 0 expects {Ws[0].shape[0]}")
    T = np.asarray(y, dtype=float)
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    if T.shape[0] != A.shape[0]:
        raise ValueError(f"geron_transfer_learning: X has {A.shape[0]} rows but y has {T.shape[0]}")
    if T.shape[1] != Ws[-1].shape[1]:
        raise ValueError(
            f"geron_transfer_learning: y has {T.shape[1]} outputs but the last layer produces {Ws[-1].shape[1]}"
        )
    L = len(Ws)
    nf = int(n_frozen)
    if not (0 <= nf < L):
        raise ValueError(f"geron_transfer_learning: n_frozen must lie in 0..{L - 1}, got {nf}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_transfer_learning: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_transfer_learning: lr must be positive and finite, got {step}")

    frozen_copy = [w.copy() for w in Ws[:nf]]
    n, d_out = T.shape

    def forward(Ws):
        acts = [A]
        H = A
        for i, w in enumerate(Ws):
            z = H @ w
            H = np.tanh(z) if i < L - 1 else z
            acts.append(H)
        return acts

    def loss_of(acts):
        diff = acts[-1] - T
        return float(np.mean(diff * diff))

    acts = forward(Ws)
    losses = [loss_of(acts)]
    for _ in range(E):
        acts = forward(Ws)
        g = 2.0 * (acts[-1] - T) / (n * d_out)
        grads = [None] * L
        for i in range(L - 1, -1, -1):
            if i < L - 1:
                g = g * (1.0 - acts[i + 1] * acts[i + 1])
            grads[i] = acts[i].T @ g
            g = g @ Ws[i].T
        for i in range(nf, L):
            Ws[i] = Ws[i] - step * grads[i]
        losses.append(loss_of(forward(Ws)))

    for i in range(nf):
        if not np.array_equal(Ws[i], frozen_copy[i]):  # invariant: frozen means frozen
            raise ValueError(f"geron_transfer_learning: internal error, frozen layer {i} changed")

    total = int(sum(w.size for w in Ws))
    trainable = int(sum(w.size for w in Ws[nf:]))

    return RichResult(
        title="Transfer learning (partial freeze)",
        summary_lines=[
            ("Layers", L),
            ("Frozen", nf),
            ("Trainable parameters", trainable),
            ("Initial loss", losses[0]),
            ("Final loss", losses[-1]),
        ],
        interpretation=(
            "Early layers hold generic features and are worth keeping; freezing them cuts the trainable "
            "parameter count, which is exactly what makes fine-tuning possible on a small new dataset."
        ),
        payload={
            "weights": Ws,
            "frozen": nf,
            "loss_curve": np.asarray(losses, dtype=float),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "trainable_params": trainable,
            "total_params": total,
            "estimate": losses[-1],
            "n": int(n),
            "method": "Fine-tuning the unfrozen tail by backpropagation; frozen layers forward-only",
        },
    )


def cheatsheet():
    return "hmtfl: Transfer learning: reuse pretrained model, fine-tune on new task"
