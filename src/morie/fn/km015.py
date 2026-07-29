# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.15: one attention head under its projections."""

import numpy as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["kamath_ch2_multihead_head_i"]


def kamath_ch2_multihead_head_i(Q, K, V, W_Qi, W_Ki, W_Vi):
    """head_i = attention(Q W_Qi, K W_Ki, V W_Vi) (row convention;
    the book's column-vector W Q is this transposed).

    Examples
    --------
    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> out = kamath_ch2_multihead_head_i(I, I, I, I, I, I)
    >>> len(out["head"]) == 2
    True

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.15, printed
    p. 36 (PDF-verified page map: printed = PDF - 27).
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    Wq = np.atleast_2d(np.asarray(W_Qi, dtype=float))
    Wk = np.atleast_2d(np.asarray(W_Ki, dtype=float))
    Wv = np.atleast_2d(np.asarray(W_Vi, dtype=float))
    for nm, X, W in (("Q", Q, Wq), ("K", K, Wk), ("V", V, Wv)):
        if X.shape[1] != W.shape[0]:
            raise ValueError(
                f"{nm} has width {X.shape[1]} but its projection has "
                f"{W.shape[0]} rows.")
    out = scaled_dot_product_attention(Q @ Wq, K @ Wk, V @ Wv)
    return RichResult(payload={
        "head": out["output"], "attention": out["attention"],
        "estimate": out["estimate"], "n": Q.shape[0],
        "method": "Single projected attention head (Kamath Eq 2.15)"})


def cheatsheet():
    return "km015: attention over per-head projections of Q, K, V"
