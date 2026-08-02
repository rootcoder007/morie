# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov's Eq 1.7: second-layer scalar output phi(W2 y1 + b21)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_layer2_output"]


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


def burkov_lm_ch1_layer2_output(W_2, y_1, b_2_1, phi="identity"):
    """y2 = phi(W2 y1 + b21), a scalar: W2 is one row in the book.

    References: Burkov LM (2025), Ch 1, Eq 1.7, p. 40.

    Examples
    --------
    >>> burkov_lm_ch1_layer2_output([1.0, -1.0], [3.0, 1.0], 0.5)["estimate"]
    2.5
    """
    w = np.atleast_1d(np.asarray(W_2, dtype=float)).ravel()
    y1 = np.atleast_1d(np.asarray(y_1, dtype=float))
    if len(w) != len(y1):
        raise ValueError(
            f"W_2 has {len(w)} weights but y_1 has {len(y1)} entries.")
    pre = float(np.dot(w, y1) + float(b_2_1))
    out = float(_phi(phi)(np.asarray(pre)))
    return RichResult(payload={
        "estimate": out, "preactivation": pre, "n": len(y1),
        "method": "Layer 2 output phi(W2 y1 + b21) (Burkov Eq 1.7)"})


def cheatsheet():
    return "b107: second-layer scalar output (Burkov Eq 1.7)"
