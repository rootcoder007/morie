# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backpropagation of errors to compute gradients."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_backpropagation"]

_ACTS = ("identity", "relu", "sigmoid", "tanh", "softmax")


def _forward_act(z, kind):
    if kind == "identity":
        return z
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "tanh":
        return np.tanh(z)
    if kind == "softmax":
        s = z - z.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)
    raise ValueError(f"geron_backpropagation: unknown activation {kind!r}; expected one of {_ACTS}")


def _act_deriv(a, z, kind):
    if kind == "identity":
        return np.ones_like(z)
    if kind == "relu":
        return (z > 0.0).astype(float)
    if kind == "sigmoid":
        return a * (1.0 - a)
    if kind == "tanh":
        return 1.0 - a * a
    raise ValueError(f"geron_backpropagation: activation {kind!r} has no elementwise derivative")


def geron_backpropagation(X, y, weights, activations, loss="mse"):
    """
    Backpropagation of errors to compute gradients.

    Formula: delta^(L) = nabla_a L * phi'(z^(L));
    delta^(l) = (W^(l+1))^T delta^(l+1) * phi'(z^(l))

    Parameters
    ----------
    X : array-like, shape (n, d0)
        Mini-batch of inputs.
    y : array-like
        Targets. Shape (n, d_L) for ``loss="mse"``; integer class indices of
        length n (or a one-hot matrix) for ``loss="ce"``.
    weights : sequence
        One entry per layer, either a weight matrix W of shape (d_in, d_out)
        or a pair ``(W, b)`` with b of length d_out.
    activations : sequence of str
        One activation name per layer, from
        ``identity, relu, sigmoid, tanh, softmax``. ``softmax`` is only
        allowed on the output layer and only with ``loss="ce"``.
    loss : {"mse", "ce"}
        Mean squared error, or mean cross-entropy against a softmax output.

    Returns
    -------
    result : RichResult
        Keys: grads_W, grads_b, deltas, loss, output, estimate, n, method.

    Examples
    --------
    A single identity layer with no bias reduces to the linear-regression
    gradient (2/n) X^T (X W - y):

    >>> r = geron_backpropagation([[1.0], [2.0]], [[1.0], [2.0]], [[[0.0]]], ["identity"])
    >>> float(r["loss"])
    2.5
    >>> float(r["grads_W"][0][0, 0])
    -5.0

    Two layers, ReLU then identity, checked against the chain rule by hand:

    >>> W1 = [[1.0, -1.0]]
    >>> W2 = [[2.0], [3.0]]
    >>> r2 = geron_backpropagation([[1.0]], [[0.0]], [W1, W2], ["relu", "identity"])
    >>> float(r2["output"][0, 0])
    2.0
    >>> [float(v) for v in r2["grads_W"][0].ravel()]
    [8.0, 0.0]

    References
    ----------
    Géron Ch 9
    """
    A0 = np.asarray(X, dtype=float)
    if A0.ndim == 1:
        A0 = A0.reshape(1, -1)
    if A0.ndim != 2:
        raise ValueError(f"geron_backpropagation: X must be 2-D, got ndim={A0.ndim}")
    n = A0.shape[0]
    if n == 0:
        raise ValueError("geron_backpropagation: X has no rows")
    if loss not in ("mse", "ce"):
        raise ValueError(f"geron_backpropagation: loss must be 'mse' or 'ce', got {loss!r}")

    Ws, bs = [], []
    for i, layer in enumerate(weights):
        if isinstance(layer, tuple) and len(layer) == 2:
            W = np.asarray(layer[0], dtype=float)
            b = np.asarray(layer[1], dtype=float).ravel()
        else:
            W = np.asarray(layer, dtype=float)
            b = np.zeros(W.shape[1] if W.ndim == 2 else 1)
        if W.ndim != 2:
            raise ValueError(f"geron_backpropagation: weights[{i}] must be a 2-D matrix, got ndim={W.ndim}")
        if b.size != W.shape[1]:
            raise ValueError(
                f"geron_backpropagation: bias {i} has {b.size} entries but layer {i} has {W.shape[1]} units"
            )
        Ws.append(W)
        bs.append(b)
    L = len(Ws)
    if L == 0:
        raise ValueError("geron_backpropagation: weights is empty")
    acts = list(activations)
    if len(acts) != L:
        raise ValueError(f"geron_backpropagation: {L} layers but {len(acts)} activations")
    for a in acts:
        if a not in _ACTS:
            raise ValueError(f"geron_backpropagation: unknown activation {a!r}; expected one of {_ACTS}")
    if "softmax" in acts[:-1]:
        raise ValueError("geron_backpropagation: softmax is only allowed on the output layer")
    if acts[-1] == "softmax" and loss != "ce":
        raise ValueError("geron_backpropagation: softmax output requires loss='ce'")
    if A0.shape[1] != Ws[0].shape[0]:
        raise ValueError(
            f"geron_backpropagation: X has {A0.shape[1]} features but layer 0 expects {Ws[0].shape[0]}"
        )
    for i in range(1, L):
        if Ws[i - 1].shape[1] != Ws[i].shape[0]:
            raise ValueError(
                f"geron_backpropagation: layer {i - 1} outputs {Ws[i - 1].shape[1]} units "
                f"but layer {i} expects {Ws[i].shape[0]}"
            )

    d_out = Ws[-1].shape[1]
    if loss == "mse":
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape != (n, d_out):
            raise ValueError(f"geron_backpropagation: y must have shape {(n, d_out)}, got {Y.shape}")
    else:
        yy = np.asarray(y)
        if yy.ndim == 2 and yy.shape == (n, d_out):
            Y = yy.astype(float)
        else:
            idx = yy.ravel().astype(int)
            if idx.size != n:
                raise ValueError(f"geron_backpropagation: y must have {n} class labels, got {idx.size}")
            if idx.min() < 0 or idx.max() >= d_out:
                raise ValueError(
                    f"geron_backpropagation: class labels must lie in [0, {d_out - 1}], got [{idx.min()}, {idx.max()}]"
                )
            Y = np.zeros((n, d_out))
            Y[np.arange(n), idx] = 1.0

    a = A0
    As = [A0]
    Zs = []
    for W, b, act in zip(Ws, bs, acts):
        z = a @ W + b
        a = _forward_act(z, act)
        Zs.append(z)
        As.append(a)
    out = As[-1]

    if loss == "mse":
        resid = out - Y
        total = float(np.sum(resid**2) / n)
        delta = (2.0 / n) * resid * _act_deriv(out, Zs[-1], acts[-1])
    else:
        if acts[-1] == "softmax":
            p = np.clip(out, 1e-15, 1.0)
            total = float(-np.sum(Y * np.log(p)) / n)
            # softmax + cross-entropy collapse to this delta exactly.
            delta = (out - Y) / n
        elif acts[-1] == "sigmoid":
            p = np.clip(out, 1e-15, 1.0 - 1e-15)
            total = float(-np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)) / n)
            delta = (out - Y) / n
        else:
            raise ValueError("geron_backpropagation: loss='ce' requires a softmax or sigmoid output activation")

    grads_W = [None] * L
    grads_b = [None] * L
    deltas = [None] * L
    for l in range(L - 1, -1, -1):
        deltas[l] = delta
        grads_W[l] = As[l].T @ delta
        grads_b[l] = delta.sum(axis=0)
        if l > 0:
            delta = (delta @ Ws[l].T) * _act_deriv(As[l], Zs[l - 1], acts[l - 1])

    return RichResult(
        title="Backpropagation",
        summary_lines=[("Loss", total), ("Layers", L), ("Batch size", n)],
        payload={
            "grads_W": grads_W,
            "grads_b": grads_b,
            "deltas": deltas,
            "loss": total,
            "output": out,
            "activations_out": As,
            "pre_activations": Zs,
            "estimate": total,
            "n": int(n),
            "method": f"Backpropagation of the {loss} loss through {L} layers",
        },
    )


def cheatsheet():
    return "hmbp: Backpropagation of errors to compute gradients"
