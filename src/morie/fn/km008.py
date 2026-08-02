# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.8: attention weights b = softmax(a)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_softmax_weights"]


def kamath_ch2_attention_softmax_weights(a):
    """b = softmax(a); sums to 1 and preserves the ordering of a.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.8, printed
    p. 32 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> out = kamath_ch2_attention_softmax_weights([0.0, 0.0])
    >>> out["weights"]
    [0.5, 0.5]
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if len(a) == 0:
        raise ValueError("no scores supplied.")
    z = a - a.max()
    b = np.exp(z) / np.exp(z).sum()
    return RichResult(payload={
        "weights": [float(v) for v in b], "estimate": float(b[0]),
        "n": len(a),
        "method": "Attention softmax weights (Kamath Eq 2.8)"})


def cheatsheet():
    return "km008: softmax over attention scores, max-shifted"
