# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: the BM25 relevance score."""

from collections import Counter

from ._richresult import RichResult

__all__ = ["kamath_bm25_score"]


def kamath_bm25_score(q_terms, doc_terms, idf, avgdl, k1=1.5, b=0.75):
    r"""BM25 = sum_t IDF(t) f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b|d|/avgdl)).

    ``idf`` is a mapping term -> IDF, or one IDF per query term in
    order. ``avgdl`` is the average document length of the collection
    and ``|d|`` is taken from ``doc_terms``. A query term absent from
    the document contributes exactly 0, which is the saturation
    formula's own answer, not a special case.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, BM25; Robertson and
    Zaragoza (2009).

    Examples
    --------
    >>> out = kamath_bm25_score(["a"], ["a", "b"], {"a": 1.0}, 2.0)
    >>> round(out["estimate"], 12)     # 1 * (1*2.5) / (1 + 1.5*1)
    1.0
    """
    q = list(q_terms)
    d = list(doc_terms)
    if len(q) == 0:
        raise ValueError("the query has no terms.")
    if len(d) == 0:
        raise ValueError("the document has no terms; |d| = 0 makes the "
                         "length normalization meaningless.")
    avg = float(avgdl)
    if avg <= 0:
        raise ValueError(f"avgdl must be positive; got {avg}.")
    k1 = float(k1)
    b = float(b)
    if k1 < 0:
        raise ValueError("k1 is a saturation parameter and cannot be "
                         "negative.")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"b must lie in [0, 1]; got {b}.")
    if hasattr(idf, "get"):
        missing = [t for t in q if t not in idf]
        if missing:
            raise ValueError(f"no IDF supplied for {missing!r}.")
        idfs = [float(idf[t]) for t in q]
    else:
        idfs = [float(v) for v in idf]
        if len(idfs) != len(q):
            raise ValueError(
                f"{len(idfs)} IDF values for {len(q)} query terms.")
    tf = Counter(d)
    norm = k1 * (1.0 - b + b * len(d) / avg)
    parts = []
    for t, w in zip(q, idfs):
        f = float(tf[t])
        parts.append(w * f * (k1 + 1.0) / (f + norm) if f > 0 else 0.0)
    return RichResult(payload={
        "estimate": float(sum(parts)), "score": float(sum(parts)),
        "per_term": parts, "term_frequencies": [int(tf[t]) for t in q],
        "doc_length": len(d), "avgdl": avg, "k1": k1, "b": b,
        "n": len(q), "method": "BM25 relevance score (Kamath Ch 7)"})


def cheatsheet():
    return "kmbm25: IDF-weighted saturated term frequency, length-normalized"
