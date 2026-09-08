# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-query attention (Shazeer 2019; Alammar Ch 3)."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["alammar_multi_query_attention"]


def alammar_multi_query_attention(Q_heads, K_shared, V_shared,
                                  n_query_heads):
    """Every query head attends over ONE shared K and V.

    References: Alammar and Grootendorst, Ch 3; Shazeer (2019).
    """
    H = int(n_query_heads)
    if H < 1:
        raise ValueError("n_query_heads must be positive.")
    if len(Q_heads) != H:
        raise ValueError(f"expected {H} query heads; got {len(Q_heads)}.")
    outs = []
    for i in range(H):
        h = scaled_dot_product_attention(Q_heads[i], K_shared, V_shared)
        outs.append(np.asarray(h["output"]))
    concat = np.concatenate(outs, axis=1)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in concat],
        "kv_cache_ratio": 1.0 / H,
        "estimate": float(concat[0, 0]), "n": H,
        "method": "Multi-query attention (Shazeer 2019)"})


def cheatsheet():
    return "almqa: all query heads share one K and one V"
