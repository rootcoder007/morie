# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multilayer perceptron forward pass."""

import numpy as np

from ._richresult import RichResult
from .grlinf import geron_linear_layer_forward

__all__ = ["geron_mlp_forward"]

_METHOD = "MLP forward pass"

_ACTS = ("relu", "tanh", "sigmoid", "identity")


def _apply(name, Z):
    if name == "relu":
        return np.maximum(Z, 0.0)
    if name == "tanh":
        return np.tanh(Z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-Z))
    return Z


def geron_mlp_forward(x, weights, biases, activation="relu", output_activation=None):
    r"""Push one instance (or a batch) through an MLP.

    .. math::
        a_0 = x,\qquad a_l = \phi(W_l a_{l-1} + b_l)\ \ (l = 1 \dots L),
        \qquad \hat y = a_L

    Each layer's affine step is delegated to
    :func:`morie.fn.grlinf.geron_linear_layer_forward`, so the
    ``(out, in)`` weight layout and its shape checks are inherited.

    ``output_activation`` defaults to ``activation``; pass
    ``"identity"`` for a regression head -- or use
    :mod:`morie.fn.grmlr` / :mod:`morie.fn.grmlc`, which are exactly
    that last layer.

    Parameters
    ----------
    x : array-like, shape (in,) or (m, in)
    weights : sequence of array-like
        ``weights[l]`` has shape ``(units_l, units_{l-1})``.
    biases : sequence of array-like
        Same length as ``weights``.
    activation : {"relu", "tanh", "sigmoid", "identity"}, optional
    output_activation : str or None, optional

    Returns
    -------
    RichResult
        Payload keys ``output``, ``activations`` (one entry per layer,
        input included), ``layer_sizes``, ``n_layers``,
        ``n_parameters``, ``dead_units`` (ReLU units at exactly zero in
        the last hidden layer), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 9, Multilayer Perceptron section.

    Examples
    --------
    Identity first layer, then a summing unit with bias ``-1``:

    >>> W = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]]]
    >>> b = [[0.0, 0.0], [-1.0]]
    >>> r = geron_mlp_forward([1.0, 2.0], W, b)
    >>> r["output"]
    [2.0]
    >>> r["layer_sizes"]
    [2, 2, 1]

    ReLU really does clip: flip the input signs and the network dies at
    the first layer.

    >>> r2 = geron_mlp_forward([-1.0, -2.0], W, b)
    >>> r2["activations"][1]
    [0.0, 0.0]
    >>> r2["output"]
    [0.0]
    """
    if activation not in _ACTS:
        raise ValueError(f"activation must be one of {_ACTS}, got {activation!r}.")
    out_act = activation if output_activation is None else output_activation
    if out_act not in _ACTS:
        raise ValueError(f"output_activation must be one of {_ACTS}, got {out_act!r}.")
    weights = list(weights)
    biases = list(biases)
    if len(weights) == 0:
        raise ValueError("weights is empty; an MLP needs at least one layer.")
    if len(weights) != len(biases):
        raise ValueError(f"got {len(weights)} weight matrices but {len(biases)} bias vectors.")

    a = np.asarray(x, dtype=float)
    batch = a.ndim == 2
    acts = [a.tolist()]
    sizes = [int(a.shape[-1])]
    n_par = 0
    L = len(weights)
    for i, (W, b) in enumerate(zip(weights, biases)):
        step = geron_linear_layer_forward(a, W, b)
        Z = np.asarray(step["output"], dtype=float)
        a = _apply(out_act if i == L - 1 else activation, Z)
        acts.append(a.tolist())
        sizes.append(step["out_features"])
        n_par += step["n_parameters"]

    last_hidden = np.asarray(acts[-2], dtype=float)
    dead = int(np.sum(last_hidden == 0.0)) if activation == "relu" and L > 1 else 0

    return RichResult(
        title="MLP forward pass",
        summary_lines=[("Layers", L), ("Sizes", sizes), ("Parameters", n_par)],
        payload={
            "output": a.tolist(),
            "activations": acts,
            "layer_sizes": sizes,
            "n_layers": L,
            "n_parameters": n_par,
            "dead_units": dead,
            "estimate": a.tolist(),
            "n": int(a.shape[0]) if batch else 1,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmlpf: a_l = phi(W_l a_{l-1} + b_l), stacked grlinf calls"
