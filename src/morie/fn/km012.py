# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.12: full scaled dot-product attention (identical to
Vaswani Eq 1; delegates to the shared core)."""

import numpy as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["kamath_ch2_scaled_dot_attention"]


def kamath_ch2_scaled_dot_attention(Q, K, V, d_k=None):
    """attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. The same
    formula as Vaswani's, so the same implementation is USED, not
    duplicated -- two copies of one formula drift.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.12, printed
    p. 33 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> out = kamath_ch2_scaled_dot_attention([[1.0, 0.0]],
    ...     [[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]])
    >>> round(out["output"][0][0], 6)
    0.669762
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    if d_k is not None and int(d_k) != Q.shape[1]:
        raise ValueError(
            f"d_k = {d_k} contradicts Q's width {Q.shape[1]}.")
    out = scaled_dot_product_attention(Q, K, V)
    return RichResult(payload={
        "output": out["output"], "attention": out["attention"],
        "estimate": out["estimate"], "n": out["n"],
        "method": "Scaled dot-product attention (Kamath Eq 2.12, "
                  "shared core)"})


def cheatsheet():
    return "km012: Eq 2.12 delegating to the shared attsdp core"
