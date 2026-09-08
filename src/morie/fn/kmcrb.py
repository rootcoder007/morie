# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: cross-encoder re-ranking."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_cross_encoder_rerank"]


def kamath_cross_encoder_rerank(q, docs, model, top_k=None):
    r"""score(q, d) = f_theta([q; SEP; d]), then sort descending.

    The point of a cross-encoder is that the query and the document
    are scored TOGETHER, so ``model`` is called once per pair -- there
    is no document-only cache to reuse. Ties keep the retrieval order
    (stable sort), so a re-rank never shuffles equal-scoring
    documents.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Cross-Encoder /
    ColBERT re-ranking; Nogueira and Cho (2019).

    Examples
    --------
    >>> out = kamath_cross_encoder_rerank("q", ["aa", "b"],
    ...                                   lambda q, d: len(d))
    >>> out["scores"], out["ranking"]
    ([2.0, 1.0], [0, 1])
    >>> out["reranked"]
    ['aa', 'b']
    """
    if not callable(model):
        raise ValueError("model must be callable model(q, d) -> score.")
    D = list(docs)
    if len(D) == 0:
        raise ValueError("there are no documents to re-rank.")
    s = np.array([float(model(q, d)) for d in D])
    if not np.all(np.isfinite(s)):
        raise ValueError("the cross-encoder returned a non-finite "
                         "score.")
    order = list(np.argsort(-s, kind="stable"))
    if top_k is not None:
        k = int(top_k)
        if not (1 <= k <= len(D)):
            raise ValueError(
                f"top_k = {k} must lie in [1, {len(D)}].")
        order = order[:k]
    return RichResult(payload={
        "estimate": float(s[order[0]]),
        "scores": [float(v) for v in s],
        "ranking": [int(i) for i in order],
        "reranked": [D[i] for i in order], "n": len(D),
        "method": "cross-encoder re-ranking (Kamath Ch 7)"})


def cheatsheet():
    return "kmcrb: joint (q, d) scoring, stable descending re-order"
