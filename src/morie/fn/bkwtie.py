# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 4: weight tying -- logits through the shared embedding."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_weight_tying"]


def burkov_weight_tying(h_last, E):
    """logits = h_L E^T with E the input embedding matrix (V x d).

    References: Burkov LM (2025), Ch 4, weight tying (Press and Wolf
    2017).

    Examples
    --------
    >>> burkov_weight_tying([1.0, 0.0], [[2.0, 0.0], [0.0, 3.0]])["logits"]
    [2.0, 0.0]
    """
    h = np.atleast_1d(np.asarray(h_last, dtype=float))
    E = np.atleast_2d(np.asarray(E, dtype=float))
    if E.shape[1] != len(h):
        raise ValueError(
            f"E is {E.shape[0]} x {E.shape[1]} but the hidden state has "
            f"{len(h)} dimensions; weight tying needs E's columns to "
            "match the hidden width.")
    logits = E @ h
    return RichResult(payload={
        "logits": [float(v) for v in logits], "estimate": float(logits[0]),
        "vocab_size": E.shape[0], "hidden_size": E.shape[1], "n": len(h),
        "method": "Weight tying logits = h E^T (Burkov Ch 4)"})


def cheatsheet():
    return "bkwtie: weight-tied output logits h E^T (Burkov Ch 4)"
