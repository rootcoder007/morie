# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head attention (Vaswani et al. 2017, Section 3.2.2)."""

import numpy as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["multi_head_attention"]


def multi_head_attention(Q, K, V, Wq, Wk, Wv, Wo, heads):
    """MHA = Concat(head_1..head_h) W_O with head_i =
    Attn(Q Wq_i, K Wk_i, V Wv_i).

    Wq, Wk, Wv are lists of per-head projection matrices; Wo maps the
    concatenation back to d_model. The head count must match the
    projection lists -- silently broadcasting one matrix over all heads
    would compute h copies of the SAME head, which is exactly the bug
    the check exists to refuse.

    References: Vaswani et al. (2017), Section 3.2.2.

    Examples
    --------
    >>> I2 = [[1.0, 0.0], [0.0, 1.0]]
    >>> out = multi_head_attention(I2, I2, I2, [I2], [I2], [I2], I2, 1)
    >>> len(out["output"]) == 2
    True
    """
    heads = int(heads)
    for name, W in (("Wq", Wq), ("Wk", Wk), ("Wv", Wv)):
        if len(W) != heads:
            raise ValueError(
                f"{name} has {len(W)} projection matrices but heads = "
                f"{heads}; one per head, or the heads are copies.")
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    outs = []
    attns = []
    for i in range(heads):
        qi = Q @ np.atleast_2d(np.asarray(Wq[i], dtype=float))
        ki = K @ np.atleast_2d(np.asarray(Wk[i], dtype=float))
        vi = V @ np.atleast_2d(np.asarray(Wv[i], dtype=float))
        h = scaled_dot_product_attention(qi, ki, vi)
        outs.append(np.asarray(h["output"]))
        attns.append(h["attention"])
    concat = np.concatenate(outs, axis=1)
    Wo = np.atleast_2d(np.asarray(Wo, dtype=float))
    if concat.shape[1] != Wo.shape[0]:
        raise ValueError(
            f"concatenated heads have width {concat.shape[1]} but Wo has "
            f"{Wo.shape[0]} rows.")
    out = concat @ Wo
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "per_head_attention": attns, "heads": heads,
        "estimate": float(out[0, 0]), "n": Q.shape[0],
        "method": "Multi-head attention (Vaswani et al. 2017)"})


def cheatsheet():
    return "attmh: Concat(head_1..head_h) W_O over per-head projections"
