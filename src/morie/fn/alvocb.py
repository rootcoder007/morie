# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tokenizer vocabulary overlap (Alammar Ch 2)."""

from ._richresult import RichResult

__all__ = ["alammar_tokenizer_vocab_overlap"]


def alammar_tokenizer_vocab_overlap(vocab_a, vocab_b):
    """Jaccard J = |A intersect B| / |A union B|.

    Examples
    --------
    >>> alammar_tokenizer_vocab_overlap(["a", "b"], ["b", "c"])["estimate"]
    0.3333333333333333
    """
    A = {str(v) for v in vocab_a}
    B = {str(v) for v in vocab_b}
    if not A and not B:
        raise ValueError("both vocabularies are empty; 0/0.")
    inter = len(A & B)
    union = len(A | B)
    return RichResult(payload={
        "estimate": inter / union, "intersection": inter, "union": union,
        "only_a": len(A - B), "only_b": len(B - A), "n": union,
        "method": "Jaccard vocabulary overlap (Alammar Ch 2)"})


def cheatsheet():
    return "alvocb: Jaccard |A&B|/|A|B| over token vocabularies"
