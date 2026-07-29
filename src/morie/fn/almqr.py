# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-query retrieval (Alammar Ch 8; RAG)."""

from ._richresult import RichResult

__all__ = ["alammar_multi_query_retrieval"]


def alammar_multi_query_retrieval(query, K, retriever, rephraser):
    """Q_set = {query} + K rephrasings; results = the UNION of top-k
    hits over the query set, first-seen order, deduplicated.

    ``rephraser`` is (query, i) -> alternative query; ``retriever`` is
    query -> ranked id list. How many documents each extra query ADDED
    is reported, since that marginal gain is the whole case for the
    technique.

    References: Alammar and Grootendorst, Ch 8.
    """
    if not callable(retriever) or not callable(rephraser):
        raise ValueError("retriever and rephraser must be callable.")
    k = int(K)
    if k < 0:
        raise ValueError("K must be non-negative.")
    queries = [str(query)] + [str(rephraser(str(query), i))
                              for i in range(k)]
    seen = []
    added = []
    for q in queries:
        hits = list(retriever(q))
        new = [h for h in hits if h not in seen]
        seen.extend(new)
        added.append(len(new))
    return RichResult(payload={
        "documents": seen, "queries": queries,
        "added_per_query": added,
        "estimate": float(len(seen)), "n": len(queries),
        "method": "Multi-query retrieval union (Alammar Ch 8)"})


def cheatsheet():
    return "almqr: union of hits over rephrasings, marginal gain per query"
