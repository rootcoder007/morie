"""Tests for bm25.bm25."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.bm25 import bm25


def _bm25_manual(docs, query, k1=1.2, b=0.75):
    """Independent reference implementation using the documented formula."""
    import math

    def tok(s):
        if isinstance(s, (list, tuple)):
            return [str(v).lower() for v in s]
        return [t for t in str(s).lower().split()]

    dl = [tok(d) for d in docs]
    N = len(dl)
    q = tok(query)
    kk = float(k1)
    bb = float(b)
    lens = [len(d) for d in dl]
    avgdl = sum(lens) / N
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
    return scores, scores_s, idf, idf_s, order, terms, avgdl, lens


def test_bm25_basic():
    """Test basic functionality against the documented formula."""
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "the quick dog jumps over the lazy fox",
        "never jump over a lazy dog twice",
    ]
    query = "quick fox"
    k1 = 1.2
    b = 0.75
    result = bm25(docs, query, k1, b)

    # Returned object exposes attributes (RichResult-like)
    assert hasattr(result, "payload")
    payload = result.payload
    assert isinstance(payload, dict)
    assert "scores" in payload
    assert "ranking" in payload
    assert "idf" in payload
    assert "terms" in payload
    assert "doc_len" in payload
    assert "avgdl" in payload
    assert "score_smooth_idf" in payload
    assert "idf_smooth" in payload

    # Independent reference computation using the same formula
    exp_scores, exp_scores_s, exp_idf, exp_idf_s, exp_ranking, exp_terms, exp_avgdl, exp_lens = _bm25_manual(
        docs, query, k1, b
    )

    assert payload["scores"] == exp_scores
    assert payload["score_smooth_idf"] == exp_scores_s
    assert payload["idf"] == exp_idf
    assert payload["idf_smooth"] == exp_idf_s
    assert payload["ranking"] == exp_ranking
    assert payload["terms"] == exp_terms
    assert payload["avgdl"] == exp_avgdl
    assert payload["doc_len"] == exp_lens
    assert payload["N"] == len(docs)
    assert payload["k1"] == k1
    assert payload["b"] == b
    assert payload["method"].startswith("Robertson")
    assert payload["estimate"] == exp_scores[0]


def test_bm25_edge():
    """Test edge cases: empty / single-doc / repeated-term queries."""
    # Single document collection still works
    docs = ["the cat sat on the mat"]
    query = "cat"
    result = bm25(docs, query)
    assert hasattr(result, "payload")
    assert len(result.payload["scores"]) == 1
    assert result.payload["N"] == 1

    # Repeated query terms are deduplicated internally to distinct terms
    docs2 = [
        "alpha beta gamma",
        "alpha alpha beta",
        "gamma gamma delta",
    ]
    query2 = "alpha alpha beta"
    result2 = bm25(docs2, query2)
    p = result2.payload
    assert isinstance(p, dict)
    assert p["terms"] == ["alpha", "beta"]
    assert len(p["idf"]) == 2
    assert len(p["scores"]) == 3
    assert len(p["ranking"]) == 3
    # ranking is a permutation of the document indices
    assert sorted(p["ranking"]) == [0, 1, 2]
    # scores are sorted in non-increasing order along the ranking
    ranked_scores = [p["scores"][i] for i in p["ranking"]]
    for a, b in zip(ranked_scores, ranked_scores[1:]):
        assert a >= b
