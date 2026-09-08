# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Okapi BM25 ranking function.

Source: Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M.
and Gatford, M. (1995), "Okapi at TREC-3", Proceedings of the Third Text
REtrieval Conference (TREC-3), NIST Special Publication 500-225,
109-126; the modern statement of the same formula is Robertson, S. and
Zaragoza, H. (2009), "The probabilistic relevance framework: BM25 and
beyond", Foundations and Trends in Information Retrieval 3(4), 333-389,
doi:10.1561/1500000019.  Neither full text was retrievable here, so the
formula is written in its standard published form:

    score(D, Q) = sum_{t in Q} IDF(t)
                  * f(t, D) (k1 + 1)
                  / ( f(t, D) + k1 (1 - b + b |D| / avgdl) )

with the Robertson-Sparck Jones inverse document frequency

    IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) ).

The two knobs are what BM25 is.  k1 controls term-frequency saturation:
at k1 = 0 the term factor collapses to 1 for any non-zero count, so the
score is just the sum of the IDFs of the query terms that appear -- a
closed form, and this module's anchor.  b controls length normalisation:
at b = 0 document length is ignored entirely, at b = 1 the count is
fully divided by relative length.

The RSJ IDF goes negative for a term appearing in more than half the
collection, which can make a document score lower for containing a query
term.  That is a real property of the formula, not a bug, and it is why
the Lucene-style smoothed variant ln(1 + (N-n+0.5)/(n+0.5)) is returned
alongside as score_smooth_idf rather than silently substituted.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["bm25"]


def tokenize(s):
    """Whitespace tokens, case folded."""
    if isinstance(s, str):
        return s.lower().split()
    return [str(v).lower() for v in s]


def bm25(docs, query, k1=1.2, b=0.75):
    """Score every document against the query.

    Parameters
    ----------
    docs : sequence
        The collection; each document a string or token list.
    query : str or sequence
        The query terms.
    k1 : float
        Term-frequency saturation, non-negative.
    b : float
        Length normalisation, in [0, 1].

    Returns
    -------
    scores : one score per document
    ranking : 0-based document indices, best first
    idf : the IDF of each distinct query term
    """
    dl = [tokenize(d) for d in docs]
    N = len(dl)
    if N == 0:
        raise ValueError("bm25: the collection is empty")
    q = tokenize(query)
    if not q:
        raise ValueError("bm25: the query is empty")
    kk = float(k1)
    if kk < 0.0:
        raise ValueError("bm25: k1 must be non-negative")
    bb = float(b)
    if not (0.0 <= bb <= 1.0):
        raise ValueError("bm25: b must lie in [0, 1]")
    lens = [len(d) for d in dl]
    tot = 0
    for v in lens:
        tot += v
    avgdl = (tot + 0.0) / N
    if avgdl <= 0.0:
        raise ValueError("bm25: every document is empty")
    tf = []
    for d in dl:
        c = {}
        for w in d:
            c[w] = c.get(w, 0) + 1
        tf.append(c)
    terms = sorted(set(q))
    idf = []
    idf_s = []
    for t in terms:
        nt = 0
        for c in tf:
            if t in c:
                nt += 1
        ratio = (N - nt + 0.5) / (nt + 0.5)
        idf.append(math.log(ratio))
        idf_s.append(math.log(1.0 + ratio))
    scores = []
    scores_s = []
    for i in range(N):
        s = 0.0
        ss = 0.0
        norm = kk * (1.0 - bb + bb * lens[i] / avgdl)
        for ti in range(len(terms)):
            f = tf[i].get(terms[ti], 0)
            if f == 0:
                continue
            w = f * (kk + 1.0) / (f + norm)
            s += idf[ti] * w
            ss += idf_s[ti] * w
        scores.append(s)
        scores_s.append(ss)
    order = sorted(range(N), key=lambda i: (-scores[i], i))
    return RichResult(
        title="Okapi BM25",
        summary_lines=[("N", N), ("terms", len(terms))],
        payload={
            "scores": scores,
            "estimate": scores[0],
            "ranking": order,
            "score_smooth_idf": scores_s,
            "idf": idf,
            "idf_smooth": idf_s,
            "terms": terms,
            "avgdl": avgdl,
            "doc_len": lens,
            "k1": kk,
            "b": bb,
            "N": N,
            "method": "Robertson et al. (1995) Okapi BM25 with Robertson-Sparck Jones IDF",
        },
    )


def cheatsheet():
    return "bm25: Okapi BM25 ranking function"
