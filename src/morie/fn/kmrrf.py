# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reciprocal Rank Fusion across several rankers."""

from ._richresult import RichResult

__all__ = ["kamath_reciprocal_rank_fusion"]


def kamath_reciprocal_rank_fusion(rankings, k=60):
    """RRF(d) = sum_r 1 / (k + rank_r(d)), ranks starting at 1.

    Only RANKS are fused, never scores, which is the whole appeal: a
    BM25 score and a cosine similarity never have to be put on the
    same scale. A document missing from a ranker contributes nothing
    from that ranker (not 1/k), and how many rankers saw each document
    is reported. The constant k damps the top of each list; 60 is the
    value from Cormack et al. (2009).

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, reciprocal rank
    fusion.

    Examples
    --------
    >>> out = kamath_reciprocal_rank_fusion([["a", "b"], ["a", "b"]])
    >>> out["ranking"]
    ['a', 'b']
    >>> abs(out["scores"]["a"] - 2 / 61) < 1e-15
    True
    >>> flip = kamath_reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    >>> abs(flip["scores"]["a"] - flip["scores"]["b"]) < 1e-15
    True
    >>> flip["n_rankers"]
    2
    """
    lists = [list(r) for r in rankings]
    k = float(k)
    if not lists:
        raise ValueError("no rankings supplied.")
    if any(not r for r in lists):
        raise ValueError(
            "a ranker returned an empty list; fusing it in silently "
            "would hide a broken retriever.")
    if k <= 0:
        raise ValueError(
            f"k must be positive; got {k}. A non-positive k can divide "
            "by zero at the top of a list.")
    scores, seen = {}, {}
    for r in lists:
        if len(set(map(repr, r))) != len(r):
            raise ValueError("a ranking contains the same document twice.")
        for pos, doc in enumerate(r, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + pos)
            seen[doc] = seen.get(doc, 0) + 1
    order = sorted(scores, key=lambda d: (-scores[d], repr(d)))
    return RichResult(payload={
        "ranking": order,
        "scores": scores,
        "appearances": seen,
        "n_rankers": len(lists),
        "n_documents": len(scores),
        "estimate": float(scores[order[0]]),
        "k": k, "n": len(scores),
        "method": "Reciprocal rank fusion sum 1/(k + rank)"})


def cheatsheet():
    return "kmrrf: sum_r 1/(k+rank_r(d)), k=60; missing doc contributes 0"
