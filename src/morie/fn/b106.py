# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.6: first hidden layer y1 = phi(W1 x + b1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_layer1_output"]


def _phi(name):
    table = {
        "identity": lambda z: z,
        "relu": lambda z: np.maximum(z, 0.0),
        "tanh": np.tanh,
        "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    }
    if callable(name):
        return name
    if name not in table:
        raise ValueError(
            f"unknown activation {name!r}; pass a callable or one of "
            f"{sorted(table)}.")
    return table[name]


def burkov_lm_ch1_layer1_output(W_1, x, b_1, phi="relu"):
    """y1 = phi(W1 x + b1).

    References: Burkov LM (2025), Ch 1, Eq 1.6, p. 39.

    Examples
    --------
    >>> burkov_lm_ch1_layer1_output([[1.0, 0.0], [0.0, -1.0]],
    ...                             [2.0, 3.0], [0.0, 0.0])["output"]
    [2.0, 0.0]
    """
    W = np.atleast_2d(np.asarray(W_1, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    b = np.atleast_1d(np.asarray(b_1, dtype=float))
    if W.shape[1] != len(x):
        raise ValueError(
            f"W_1 has {W.shape[1]} columns but x has {len(x)} entries.")
    if W.shape[0] != len(b):
        raise ValueError(
            f"W_1 has {W.shape[0]} rows but b_1 has {len(b)} entries.")
    pre = W @ x + b
    out = _phi(phi)(pre)
    return RichResult(payload={
        "output": [float(v) for v in out],
        "preactivation": [float(v) for v in pre],
        "estimate": float(out[0]), "n": len(out),
        "method": "Layer 1 output phi(W1 x + b1) (Burkov Eq 1.6)"})


def cheatsheet():
    return "b106: first hidden layer phi(W1 x + b1) (Burkov Eq 1.6)"
