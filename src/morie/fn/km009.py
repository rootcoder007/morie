# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.9: one softmax element written out."""

from . import _array_core as np

from ._richresult import RichResult
from .km008 import kamath_ch2_attention_softmax_weights

__all__ = ["kamath_ch2_softmax_element"]


def kamath_ch2_softmax_element(a_i, a):
    """b_i = exp(a_i) / sum_j exp(a_j); a_i must be one of the scores
    in a, and the element is checked against the full Eq 2.8 vector.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.9, printed
    p. 32 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_softmax_element(0.0, [0.0, 0.0])["estimate"]
    0.5
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    ai = float(a_i)
    matches = np.flatnonzero(np.isclose(a, ai))
    if len(matches) == 0:
        raise ValueError(
            "a_i is not one of the scores in a; Eq 2.9 is an element of "
            "Eq 2.8's vector, not a free function.")
    full = kamath_ch2_attention_softmax_weights(a)["weights"]
    b_i = full[int(matches[0])]
    return RichResult(payload={
        "estimate": float(b_i), "index": int(matches[0]),
        "full_weights": full, "n": len(a),
        "method": "Softmax element (Kamath Eq 2.9)"})


def cheatsheet():
    return "km009: single softmax element, pinned to the Eq 2.8 vector"
