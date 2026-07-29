# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: add-k smoothing, the generalised Laplace."""

from ._richresult import RichResult

__all__ = ["burkov_add_k_smoothing"]


def burkov_add_k_smoothing(counts_ngram, counts_prefix, V, k=0.5):
    """P = (count + k) / (prefix + kV); k = 1 recovers Laplace.

    References: Burkov LM (2025), Ch 2, add-k smoothing.

    Examples
    --------
    >>> burkov_add_k_smoothing(0, 0, 4, k=0.5)["estimate"]
    0.25
    """
    c = float(counts_ngram); p = float(counts_prefix)
    v = int(V); k = float(k)
    if c < 0 or p < 0:
        raise ValueError("counts must be non-negative.")
    if v < 1:
        raise ValueError(f"vocabulary size must be positive; got {V}.")
    if k <= 0:
        raise ValueError(
            f"k must be positive; got {k}. k = 0 is the unsmoothed MLE "
            "and has its own function.")
    if c > p:
        raise ValueError("count(ngram) cannot exceed count(prefix).")
    est = (c + k) / (p + k * v)
    return RichResult(payload={
        "estimate": est, "count_ngram": c, "count_prefix": p,
        "vocab_size": v, "k": k, "n": int(p),
        "method": "Add-k smoothing (Burkov Ch 2)"})


def cheatsheet():
    return "bkaddk: add-k smoothing (count+k)/(prefix+kV) (Burkov Ch 2)"
