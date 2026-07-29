# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.19: masked attention with an additive mask INSIDE
the scaling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_masked_attention"]


def kamath_ch2_masked_attention(Q, K, V, M, d_k=None):
    """maskedAttention = softmax((QK^T + M) / sqrt(d_k)) V.

    NOTE the book puts the mask inside the scaling -- (QK^T + M) is
    divided by sqrt(d_k) -- where Vaswani's convention adds the mask
    after. For -inf masks the two agree; for FINITE masks they do
    not, and the tests pin the difference rather than papering over
    it.

    Examples
    --------
    >>> out = kamath_ch2_masked_attention([[1.0, 0.0]],
    ...     [[1.0, 0.0], [0.0, 1.0]], [[7.0], [0.0]],
    ...     [[0.0, float("-inf")]])
    >>> out["attention"][0]
    [1.0, 0.0]

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.19, printed
    p. 38 (PDF-verified page map: printed = PDF - 27).
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must share d_k.")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same number of rows.")
    d = Q.shape[1] if d_k is None else int(d_k)
    if d != Q.shape[1]:
        raise ValueError(f"d_k = {d} contradicts Q's width "
                         f"{Q.shape[1]}.")
    scores = (Q @ K.T + M) / np.sqrt(d)
    if M.shape != scores.shape:
        raise ValueError("M's shape must match QK^T.")
    z = scores - scores.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        A = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    out = A @ V
    return RichResult(payload={
        "output": [[float(v) for v in r] for r in out],
        "attention": [[float(v) for v in r] for r in A],
        "estimate": float(out[0, 0]), "n": Q.shape[0],
        "method": "Masked attention, mask inside the scaling "
                  "(Kamath Eq 2.19)"})


def cheatsheet():
    return "km019: softmax((QK^T + M)/sqrt(d_k))V, book's mask placement"
