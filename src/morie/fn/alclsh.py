# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification head on the [CLS] vector (Alammar Ch 4)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_classification_head"]


def alammar_classification_head(h_cls, W_cls, b):
    """logits = W h_CLS + b; probabilities by softmax; argmax class.

    References: Alammar and Grootendorst, Ch 4.

    Examples
    --------
    >>> out = alammar_classification_head([1.0, 0.0],
    ...     [[2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    >>> out["predicted_class"]
    0
    """
    h = np.atleast_1d(np.asarray(h_cls, dtype=float))
    W = np.atleast_2d(np.asarray(W_cls, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if W.shape[1] != len(h):
        raise ValueError(
            f"W has {W.shape[1]} columns but h_cls has {len(h)} entries.")
    if W.shape[0] != len(b):
        raise ValueError(
            f"W has {W.shape[0]} rows but b has {len(b)} entries.")
    logits = W @ h + b
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    return RichResult(payload={
        "logits": [float(v) for v in logits],
        "probabilities": [float(v) for v in p],
        "predicted_class": int(np.argmax(logits)),
        "estimate": float(logits[0]), "n": len(b),
        "method": "Linear classification head + softmax (Alammar Ch 4)"})


def cheatsheet():
    return "alclsh: softmax(W h_CLS + b), argmax class"
