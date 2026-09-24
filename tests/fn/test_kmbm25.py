"""Verification tests for kmbm25.

Kamath, Keenan, Somers and Sorenson (2024), chapter 7, the BM25 relevance score. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.kmbm25 import kamath_bm25_score


def test_bm25_matches_the_saturating_term_weight():
    # BM25 = sum_t idf(t) f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b |d|/avgdl))
    q = ["cat", "sat"]
    doc = ["cat", "cat", "mat"]
    idf = {"cat": 1.2, "sat": 0.8}
    avgdl, k1, b = 3.0, 1.5, 0.75
    res = kamath_bm25_score(q, doc, idf, avgdl, k1=k1, b=b)
    total = 0.0
    for t in q:
        f = doc.count(t)
        if f:
            total += idf[t] * f * (k1 + 1.0) / (
                f + k1 * (1.0 - b + b * len(doc) / avgdl))
    assert res["estimate"] == pytest.approx(total, rel=1e-12)
    assert res["doc_length"] == len(doc)


def test_a_term_absent_from_the_document_contributes_nothing():
    res = kamath_bm25_score(["zebra"], ["cat", "mat"], {"zebra": 2.0}, 2.0)
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_term_frequency_saturates_rather_than_growing_without_bound():
    # the k1 denominator caps the gain from repetition: doubling the
    # count does not double the score
    idf = {"cat": 1.0}
    one = kamath_bm25_score(["cat"], ["cat"], idf, 1.0)["estimate"]
    four = kamath_bm25_score(["cat"], ["cat"] * 4, idf, 4.0)["estimate"]
    assert four < 4.0 * one


def test_a_longer_document_is_penalised_at_equal_term_frequency():
    idf = {"cat": 1.0}
    short = kamath_bm25_score(["cat"], ["cat", "a"], idf, 4.0)["estimate"]
    long_ = kamath_bm25_score(["cat"], ["cat"] + ["a"] * 7, idf, 4.0)["estimate"]
    assert long_ < short
