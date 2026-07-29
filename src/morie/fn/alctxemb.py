# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contextualised embedding extraction (Alammar Ch 2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_contextualized_embedding"]


def alammar_contextualized_embedding(layer_outputs, layer_idx, position):
    """h_position^(layer): pick one vector out of the stack, with the
    layer convention stated -- index 0 is the embedding layer, -1 the
    last. What makes an embedding CONTEXTUAL is that the same token at
    another position may have a different vector, and the payload
    reports whether that holds in the supplied stack.

    References: Alammar and Grootendorst, Ch 2.
    """
    L = np.asarray(layer_outputs, dtype=float)
    if L.ndim != 3:
        raise ValueError(
            "layer_outputs must be (n_layers, seq_len, dim); got shape "
            f"{L.shape}.")
    li = int(layer_idx); pos = int(position)
    n_layers, seq, dim = L.shape
    if not -n_layers <= li < n_layers:
        raise ValueError(f"layer {li} out of range for {n_layers} layers.")
    if not -seq <= pos < seq:
        raise ValueError(f"position {pos} out of range for length {seq}.")
    v = L[li, pos]
    varies = bool(seq > 1 and not np.allclose(L[li], L[li, 0]))
    return RichResult(payload={
        "embedding": [float(x) for x in v],
        "context_varies": varies,
        "estimate": float(v[0]), "layer": li, "position": pos, "n": seq,
        "method": "Contextualised embedding extraction (Alammar Ch 2)"})


def cheatsheet():
    return "alctxemb: h_pos^(layer) from a (layers, seq, dim) stack"
