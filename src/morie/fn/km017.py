# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.17: the position-wise feed-forward block."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_ffn_relu"]


def kamath_ch2_ffn_relu(z, W_1, W_2, b_1, b_2):
    """F(z) = ReLU(z W_1 + b_1) W_2 + b_2, applied position-wise.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.17, printed
    p. 37 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_ffn_relu([[1.0]], [[-2.0]], [[3.0]], [0.0],
    ...                     [0.5])["output"]
    [[0.5]]
    """
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    W1 = np.atleast_2d(np.asarray(W_1, dtype=float))
    W2 = np.atleast_2d(np.asarray(W_2, dtype=float))
    b1 = np.atleast_1d(np.asarray(b_1, dtype=float))
    b2 = np.atleast_1d(np.asarray(b_2, dtype=float))
    if Z.shape[1] != W1.shape[0]:
        raise ValueError("z's width must match W_1's rows.")
    if W1.shape[1] != len(b1):
        raise ValueError("b_1 must match W_1's columns.")
    if W1.shape[1] != W2.shape[0]:
        raise ValueError("W_2's rows must match W_1's columns.")
    if W2.shape[1] != len(b2):
        raise ValueError("b_2 must match W_2's columns.")
    hidden = np.maximum(Z @ W1 + b1, 0.0)
    out = hidden @ W2 + b2
    return RichResult(payload={
        "output": [[float(v) for v in r] for r in out],
        "hidden": [[float(v) for v in r] for r in hidden],
        "estimate": float(out[0, 0]), "n": Z.shape[0],
        "method": "Position-wise FFN ReLU(zW1+b1)W2+b2 "
                  "(Kamath Eq 2.17)"})


def cheatsheet():
    return "km017: two-layer position-wise FFN with ReLU"
