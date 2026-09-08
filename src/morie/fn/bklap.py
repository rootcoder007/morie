# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 2: add-1 (Laplace) smoothing."""

from ._richresult import RichResult

__all__ = ["burkov_laplace_add_one"]


def burkov_laplace_add_one(counts_ngram, counts_prefix, V):
    """P = (count + 1) / (prefix + V).

    References: Burkov LM (2025), Ch 2, Laplace smoothing.

    Examples
    --------
    >>> burkov_laplace_add_one(0, 0, 4)["estimate"]
    0.25
    """
    c = float(counts_ngram); p = float(counts_prefix); v = int(V)
    if c < 0 or p < 0:
        raise ValueError("counts must be non-negative.")
    if v < 1:
        raise ValueError(f"vocabulary size must be positive; got {V}.")
    if c > p:
        raise ValueError("count(ngram) cannot exceed count(prefix).")
    est = (c + 1.0) / (p + v)
    return RichResult(payload={
        "estimate": est, "count_ngram": c, "count_prefix": p,
        "vocab_size": v, "n": int(p),
        "method": "Laplace add-1 smoothing (Burkov Ch 2)"})


def cheatsheet():
    return "bklap: Laplace add-1 smoothing (count+1)/(prefix+V) (Burkov Ch 2)"
