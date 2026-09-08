# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recall@k (Alammar Ch 8)."""

from ._richresult import RichResult

__all__ = ["alammar_recall_at_k"]


def alammar_recall_at_k(retrieved, relevant, k):
    """Recall@k = |relevant intersect top-k| / |relevant|.

    Examples
    --------
    >>> alammar_recall_at_k([1, 2, 3, 4], [2, 9], 3)["estimate"]
    0.5
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive.")
    rel = set(relevant)
    if not rel:
        raise ValueError(
            "the relevant set is empty; recall is 0/0 and reporting 1 "
            "or 0 there would be a choice, not a measurement.")
    top = list(retrieved)[:k]
    hit = len(rel & set(top))
    return RichResult(payload={
        "estimate": hit / len(rel), "hits": hit,
        "n_relevant": len(rel), "k": k, "n": len(top),
        "method": "Recall@k (Alammar Ch 8)"})


def cheatsheet():
    return "alrck: |relevant in top-k| / |relevant|, empty set refused"
