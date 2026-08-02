# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KV-cache append + single-step attention (Alammar Ch 3)."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["alammar_kv_cache_lookup"]


def alammar_kv_cache_lookup(K_cache, V_cache, k_new, v_new, q_new):
    """K_t = [K_{t-1}; k_t]; V_t = [V_{t-1}; v_t]; one attention row.

    The point of the cache is that the step is O(t), not O(t^2): only
    the new query attends, over all cached keys. The tests verify the
    result equals full attention's last row recomputed from scratch.

    References: Alammar and Grootendorst, Ch 3 (KV cache).
    """
    k_new = np.atleast_2d(np.asarray(k_new, dtype=float))
    v_new = np.atleast_2d(np.asarray(v_new, dtype=float))
    q_new = np.atleast_2d(np.asarray(q_new, dtype=float))
    if K_cache is None or len(np.atleast_2d(K_cache)) == 0:
        K = k_new; V = v_new
    else:
        K = np.vstack([np.atleast_2d(np.asarray(K_cache, dtype=float)),
                       k_new])
        V = np.vstack([np.atleast_2d(np.asarray(V_cache, dtype=float)),
                       v_new])
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V caches must stay the same length.")
    out = scaled_dot_product_attention(q_new, K, V)
    return RichResult(payload={
        "output": out["output"][0], "attention": out["attention"][0],
        "K_cache": [[float(v) for v in r] for r in K],
        "V_cache": [[float(v) for v in r] for r in V],
        "cache_length": K.shape[0],
        "estimate": out["estimate"], "n": K.shape[0],
        "method": "KV-cache single-step attention (Alammar Ch 3)"})


def cheatsheet():
    return "alkvc2: append k,v to the cache, attend with one query row"
