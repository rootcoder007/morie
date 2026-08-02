# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sliding-window attention (Beltagy et al. 2020; Alammar Ch 3)."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["alammar_sliding_window_attention"]


def alammar_sliding_window_attention(Q, K, V, window_size):
    """Causal window: position i attends to j in [i - W + 1, i].

    Every row still sums to 1 -- position 0 sees only itself. The
    tests assert zero attention outside the band, which a full-window
    stub cannot fake.

    References: Alammar and Grootendorst, Ch 3; Beltagy et al. (2020).
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    W = int(window_size)
    if W < 1:
        raise ValueError(f"window_size must be positive; got "
                         f"{window_size}.")
    n, m = Q.shape[0], K.shape[0]
    if n != m:
        raise ValueError(
            "sliding-window attention is defined over one sequence; "
            f"got {n} queries and {m} keys.")
    mask = np.full((n, m), -np.inf)
    for i in range(n):
        lo = max(0, i - W + 1)
        mask[i, lo:i + 1] = 0.0
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    return RichResult(payload={
        "output": out["output"], "attention": out["attention"],
        "window": W, "estimate": out["estimate"], "n": n,
        "method": "Sliding-window causal attention (Beltagy et al. 2020)"})


def cheatsheet():
    return "alswa: causal attention restricted to the last W positions"
