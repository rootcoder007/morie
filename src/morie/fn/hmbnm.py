# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Biological neuron model (McCulloch-Pitts): weighted sum then activation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_biological_neuron"]


def _activate(z, kind):
    if kind == "step":
        return np.where(z >= 0.0, 1.0, 0.0)
    if kind == "sign":
        return np.where(z >= 0.0, 1.0, -1.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "identity":
        return z
    raise ValueError(
        f"geron_biological_neuron: unknown activation {kind!r}; "
        "expected one of step, sign, sigmoid, tanh, relu, identity"
    )


def geron_biological_neuron(x, w, b, activation="step"):
    """
    Biological neuron model (McCulloch-Pitts): weighted sum then activation.

    Formula: a = phi(sum_i w_i x_i + b)

    Parameters
    ----------
    x : array-like
        Input vector of length d, or a stack of inputs with shape (n, d).
    w : array-like
        Weight vector of length d.
    b : float
        Bias (the negated firing threshold).
    activation : {"step", "sign", "sigmoid", "tanh", "relu", "identity"}
        Activation function phi. The original McCulloch-Pitts neuron uses
        the Heaviside step.

    Returns
    -------
    result : RichResult
        Keys: a, z, fires, estimate, n, method.

    Examples
    --------
    >>> r = geron_biological_neuron([1.0, 2.0], [0.5, -1.0], 0.25)
    >>> float(r["z"])
    -1.25
    >>> float(r["a"])
    0.0
    >>> r2 = geron_biological_neuron([1.0, 2.0], [0.5, -1.0], 1.5)
    >>> float(r2["z"]), float(r2["a"])
    (0.0, 1.0)
    >>> round(float(geron_biological_neuron([1.0], [1.0], 0.0, activation="sigmoid")["a"]), 6)
    0.731059

    References
    ----------
    Géron Ch 9
    """
    xs = np.asarray(x, dtype=float)
    ws = np.asarray(w, dtype=float).ravel()
    if ws.size == 0:
        raise ValueError("geron_biological_neuron: w is empty")
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)
    elif xs.ndim != 2:
        raise ValueError(f"geron_biological_neuron: x must be 1-D or 2-D, got ndim={xs.ndim}")
    if xs.shape[1] != ws.size:
        raise ValueError(
            f"geron_biological_neuron: x has {xs.shape[1]} features but w has {ws.size} weights"
        )
    bb = float(np.asarray(b, dtype=float).ravel()[0]) if np.size(b) else 0.0
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ws)):
        raise ValueError("geron_biological_neuron: x and w must be finite")

    z = xs @ ws + bb
    a = _activate(z, activation)
    fires = np.asarray(z >= 0.0)

    scalar = z.size == 1
    return RichResult(
        title="McCulloch-Pitts neuron",
        summary_lines=[("Net input z", float(z[0])), ("Activation", activation)],
        payload={
            "a": float(a[0]) if scalar else a,
            "z": float(z[0]) if scalar else z,
            "fires": bool(fires[0]) if scalar else fires,
            "threshold": -bb,
            "activation": activation,
            "estimate": float(a[0]),
            "n": int(z.size),
            "method": "McCulloch-Pitts neuron a = phi(w.x + b)",
        },
    )


def cheatsheet():
    return "hmbnm: Biological neuron model (McCulloch-Pitts): weighted sum then activation"
