# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.40: top-k gating G(x) = softmax(TopK(x W_g))."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_moe_topk_gating"]


def kamath_ch2_moe_topk_gating(x, W_g, k=2):
    """Scores x W_g; all but the top k set to -inf BEFORE the softmax,
    so the surviving weights renormalise over the selected experts and
    exactly n - k weights are 0.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.40, printed
    p. 74.

    Examples
    --------
    >>> out = kamath_ch2_moe_topk_gating([1.0], [[3.0, 1.0, 2.0]], k=2)
    >>> out["n_active"]
    2
    >>> out["weights"][1]
    0.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    W = np.atleast_2d(np.asarray(W_g, dtype=float))
    if W.shape[0] != len(x):
        raise ValueError(
            f"W_g has {W.shape[0]} rows but x has {len(x)} "
            "dimensions.")
    k = int(k)
    n = W.shape[1]
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}].")
    scores = x @ W
    order = np.argsort(-scores)
    masked = np.full(n, -np.inf)
    masked[order[:k]] = scores[order[:k]]
    z = masked - masked.max()
    with np.errstate(invalid="ignore"):
        e = np.exp(z)
    w = e / e.sum()
    return RichResult(payload={
        "weights": [float(v) for v in w],
        "selected_experts": [int(i) for i in sorted(order[:k])],
        "n_active": int(np.sum(w > 0)),
        "estimate": float(w.max()), "n": n,
        "method": "Top-k expert gating (Kamath Eq 2.40)"})


def cheatsheet():
    return "km040: softmax over the top-k gate scores, rest exactly 0"
