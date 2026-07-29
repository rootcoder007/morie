# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.7: the generic attention score a_i = alpha(q, k_i)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_attention_score"]


def kamath_ch2_attention_score(q, k_i, alpha="scaled_dot"):
    """a_i = alpha(q, k_i); alpha is "dot", "scaled_dot", "cosine" or
    a callable. Eq 2.7 deliberately leaves alpha abstract; the named
    options are the ones the chapter then instantiates.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.7, printed
    p. 32 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> kamath_ch2_attention_score([1.0, 0.0], [1.0, 0.0], "dot")["estimate"]
    1.0
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    k = np.atleast_1d(np.asarray(k_i, dtype=float))
    if q.shape != k.shape:
        raise ValueError("q and k_i must have the same dimension.")
    if callable(alpha):
        a = float(alpha(q, k)); name = "callable"
    elif alpha == "dot":
        a = float(np.dot(q, k)); name = "dot"
    elif alpha == "scaled_dot":
        a = float(np.dot(q, k) / np.sqrt(len(q))); name = "scaled_dot"
    elif alpha == "cosine":
        nq, nk = np.linalg.norm(q), np.linalg.norm(k)
        if nq == 0 or nk == 0:
            raise ValueError("cosine score is undefined for a zero "
                             "vector.")
        a = float(np.dot(q, k) / (nq * nk)); name = "cosine"
    else:
        raise ValueError(f"unknown alpha {alpha!r}.")
    return RichResult(payload={
        "estimate": a, "alpha": name, "n": len(q),
        "method": "Attention score a_i = alpha(q, k_i) (Kamath Eq 2.7)"})


def cheatsheet():
    return "km007: pluggable attention score, scaled dot default"
