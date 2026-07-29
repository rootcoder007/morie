# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.10: the attention output o = sum b_i v_i."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_output"]


def kamath_ch2_attention_output(b, v):
    """o = sum_i b_i v_i -- a convex combination when b comes from a
    softmax, which the payload checks and reports.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.10, printed
    p. 32 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_attention_output([0.5, 0.5], [[2.0], [4.0]])["output"]
    [3.0]
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    V = np.atleast_2d(np.asarray(v, dtype=float))
    if V.shape[0] != len(b):
        raise ValueError(
            f"need one value row per weight; got {V.shape[0]} rows for "
            f"{len(b)} weights.")
    o = b @ V
    convex = bool(np.all(b >= 0) and abs(float(b.sum()) - 1.0) < 1e-9)
    return RichResult(payload={
        "output": [float(x) for x in o], "is_convex_combination": convex,
        "estimate": float(o[0]), "n": len(b),
        "method": "Attention output sum b_i v_i (Kamath Eq 2.10)"})


def cheatsheet():
    return "km010: weighted value sum, convexity reported"
