# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification MLP: softmax output and cross-entropy loss."""

from . import _array_core as np

from ._richresult import RichResult
from .hmcec import geron_cross_entropy_cost

__all__ = ["geron_classification_mlp", "mlp_init"]


def mlp_init(sizes, seed=0, scale=None):
    """Deterministic He-style initialisation from the LCG, plus zero biases."""
    s = int(seed) % 2**32
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        sd = float(np.sqrt(2.0 / fan_in)) if scale is None else float(scale)
        n = fan_in * fan_out
        u = np.empty(n)
        for k in range(n):
            s = (1664525 * s + 1013904223) % 2**32
            u[k] = (s + 0.5) / 2**32
        # Uniform on [-sqrt(3) sd, sqrt(3) sd] has standard deviation sd.
        W = (2.0 * u - 1.0) * np.sqrt(3.0) * sd
        Ws.append(W.reshape(fan_in, fan_out))
        bs.append(np.zeros(fan_out))
    return Ws, bs


def geron_classification_mlp(X, y, hidden_sizes=(4,), epochs=100, lr=0.1, seed=0):
    """
    Classification MLP: softmax output and cross-entropy loss.

    Formula: loss = -sum_k y_k log(softmax(logits)_k)

    A full ReLU multilayer perceptron trained by real backpropagation:
    forward pass, analytic gradients, full-batch gradient descent. The
    output layer's gradient is the one identity worth remembering --
    ``dL/dlogits = (P - Y)/m``, softmax and cross-entropy collapsing into
    a subtraction -- and every hidden layer then applies
    ``delta <- (delta W^T) * [z > 0]``.

    The loss itself is DELEGATED to
    :func:`morie.fn.hmcec.geron_cross_entropy_cost` so the reported number
    is computed by the same code path as the standalone cost function.

    Weights start from a deterministic LCG draw scaled He-style
    (``sd = sqrt(2/fan_in)``), biases at zero, so with zero-mean inputs
    the initial loss sits at about ``log K`` and every run reproduces.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Integer class labels.
    hidden_sizes : sequence of int, default (4,)
        Hidden layer widths; may be empty for a plain softmax regression.
    epochs : int, default 100
    lr : float, default 0.1
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: weights, biases, loss_history, predictions, probabilities,
        accuracy, n_params, layer_sizes, estimate, n, method.

    Examples
    --------
    A linearly separable two-class problem is learned to zero error, and
    the loss falls monotonically:

    >>> X = [[-2.0], [-1.0], [1.0], [2.0]]
    >>> y = [0, 0, 1, 1]
    >>> r = geron_classification_mlp(X, y, hidden_sizes=(4,), epochs=300, lr=0.5, seed=1)
    >>> r["accuracy"]
    1.0
    >>> r["loss_history"][-1] < r["loss_history"][0]
    True
    >>> r["predictions"]
    [0, 0, 1, 1]

    The parameter count is exact: 1->4 is 8 parameters, 4->2 is 10.

    >>> r["n_params"], r["layer_sizes"]
    (18, [1, 4, 2])

    Untrained, the network sits near chance, ``log 2``:

    >>> import math
    >>> r0 = geron_classification_mlp(X, y, hidden_sizes=(4,), epochs=1, lr=0.0, seed=1)
    >>> abs(r0["loss_history"][0] - math.log(2)) < 0.2
    True

    XOR needs the hidden layer -- with none, the model is a linear
    classifier and cannot get all four corners right:

    >>> Xx = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> yx = [0, 1, 1, 0]
    >>> geron_classification_mlp(Xx, yx, hidden_sizes=(), epochs=400, lr=0.5)["accuracy"]
    0.75

    References
    ----------
    Géron Ch 9
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y).ravel()
    if Xa.size == 0:
        raise ValueError("geron_classification_mlp: X is empty")
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_classification_mlp: y has {ya.size} entries but X has {Xa.shape[0]} rows")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_classification_mlp: X contains non-finite values")
    labels = np.unique(ya)
    K = int(labels.size)
    if K < 2:
        raise ValueError(f"geron_classification_mlp: need at least 2 classes, got {K}")
    idx = np.searchsorted(labels, ya)
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_classification_mlp: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if eta < 0 or not np.isfinite(eta):
        raise ValueError(f"geron_classification_mlp: lr must be non-negative and finite, got {lr!r}")
    hs = [int(h) for h in hidden_sizes]
    if any(h < 1 for h in hs):
        raise ValueError(f"geron_classification_mlp: hidden sizes must be >= 1, got {hidden_sizes!r}")

    m, n = Xa.shape
    sizes = [n] + hs + [K]
    Ws, bs = mlp_init(sizes, seed=seed)
    Y = np.zeros((m, K))
    Y[np.arange(m), idx] = 1.0

    hist = []
    for _ in range(E):
        acts = [Xa]
        zs = []
        h = Xa
        for i in range(len(Ws) - 1):
            z = h @ Ws[i] + bs[i]
            zs.append(z)
            h = np.maximum(z, 0.0)
            acts.append(h)
        logits = h @ Ws[-1] + bs[-1]
        shift = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(shift)
        P /= P.sum(axis=1, keepdims=True)
        hist.append(float(np.mean(-np.sum(Y * (shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))), axis=1))))

        delta = (P - Y) / m
        for i in range(len(Ws) - 1, -1, -1):
            gW = acts[i].T @ delta
            gb = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ Ws[i].T) * (zs[i - 1] > 0)
            Ws[i] = Ws[i] - eta * gW
            bs[i] = bs[i] - eta * gb

    h = Xa
    for i in range(len(Ws) - 1):
        h = np.maximum(h @ Ws[i] + bs[i], 0.0)
    logits = h @ Ws[-1] + bs[-1]
    final = geron_cross_entropy_cost(np.hstack([h, np.ones((m, 1))]), Y, np.vstack([Ws[-1], bs[-1][None, :]]))
    P = np.asarray(final["probabilities"], dtype=float)
    pred = labels[P.argmax(axis=1)]
    acc = float(np.mean(pred == ya))
    n_params = int(sum(W.size for W in Ws) + sum(b.size for b in bs))

    return RichResult(
        title="Classification MLP",
        summary_lines=[("Accuracy", acc), ("Final loss", float(final["cost"])), ("Parameters", n_params)],
        interpretation="Softmax + cross-entropy gives dL/dlogits = (P - Y)/m; every hidden layer just chains a ReLU mask.",
        payload={
            "weights": [W.tolist() for W in Ws],
            "biases": [b.tolist() for b in bs],
            "loss_history": hist,
            "final_loss": float(final["cost"]),
            "predictions": pred.tolist(),
            "probabilities": P.tolist(),
            "logits": logits.tolist(),
            "accuracy": acc,
            "classes": labels.tolist(),
            "layer_sizes": sizes,
            "n_params": n_params,
            "estimate": acc,
            "n": int(m),
            "method": "ReLU MLP trained by backpropagation; loss delegated to hmcec",
        },
    )


def cheatsheet():
    return "hmclsn: Classification MLP: softmax output and cross-entropy loss"
