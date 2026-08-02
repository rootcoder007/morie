# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grouped-query attention (Ainslie et al. 2023; Alammar Ch 3)."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["alammar_grouped_query_attention"]


def alammar_grouped_query_attention(Q_heads, K_groups, V_groups,
                                    n_query_heads, n_kv_groups):
    """head_i = Attn(Q_i, K_{g(i)}, V_{g(i)}) with g(i) = i mod G.

    G = 1 recovers multi-query attention; G = H recovers full
    multi-head. H must be divisible by G, and the tests pin both
    limiting cases against the plain implementations.

    References: Alammar and Grootendorst, Ch 3 (GQA); Ainslie et al.
    (2023).
    """
    H = int(n_query_heads); G = int(n_kv_groups)
    if H < 1 or G < 1:
        raise ValueError("head and group counts must be positive.")
    if H % G != 0:
        raise ValueError(
            f"n_query_heads = {H} must be divisible by n_kv_groups = {G}.")
    if len(Q_heads) != H:
        raise ValueError(f"expected {H} query heads; got {len(Q_heads)}.")
    if len(K_groups) != G or len(V_groups) != G:
        raise ValueError(f"expected {G} K and V groups; got "
                         f"{len(K_groups)} and {len(V_groups)}.")
    outs = []
    assignment = []
    for i in range(H):
        g = i % G
        assignment.append(g)
        h = scaled_dot_product_attention(Q_heads[i], K_groups[g],
                                         V_groups[g])
        outs.append(np.asarray(h["output"]))
    concat = np.concatenate(outs, axis=1)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in concat],
        "group_assignment": assignment,
        "kv_cache_ratio": G / H,
        "estimate": float(concat[0, 0]), "n": H,
        "method": "Grouped-query attention (Ainslie et al. 2023)"})


def cheatsheet():
    return "algqa: per-head attention over G shared KV groups, g(i) = i mod G"
