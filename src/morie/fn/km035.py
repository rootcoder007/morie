# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.35: GPT's supervised softmax head."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt_supervised_softmax"]


def kamath_ch2_gpt_supervised_softmax(x, h, W_y):
    """P(y | x) = softmax(h_m^l W_y): the final hidden state through
    the label projection. ``x`` is recorded for the signature; the
    computation reads h and W_y.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.35, printed
    p. 70.

    Examples
    --------
    >>> out = kamath_ch2_gpt_supervised_softmax("doc", [1.0, 0.0],
    ...     [[2.0, 0.0], [0.0, 2.0]])
    >>> out["predicted_class"]
    0
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    W = np.atleast_2d(np.asarray(W_y, dtype=float))
    if W.shape[1] != len(h):
        raise ValueError(
            f"W_y has {W.shape[1]} columns but h has {len(h)} "
            "dimensions (row convention h W_y^T).")
    logits = W @ h
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    return RichResult(payload={
        "probabilities": [float(v) for v in p],
        "predicted_class": int(np.argmax(p)),
        "estimate": float(p.max()), "n": len(p),
        "method": "GPT supervised softmax head (Kamath Eq 2.35)"})


def cheatsheet():
    return "km035: softmax(h W_y) over task labels"
