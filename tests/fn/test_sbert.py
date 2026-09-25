"""Tests for sbert.sbert (bi-encoder cosine scores, Reimers & Gurevych 2019)."""

import math

import pytest

from morie.fn.sbert import sbert

VEC = {"a cat": [1.0, 0.2, 0.0], "a dog": [0.9, 0.4, 0.1], "stocks fell": [-0.2, 0.1, 1.0]}


def test_sbert_basic():
    """Cosine of the two embeddings, recomputed; each distinct sentence
    is embedded exactly once however often it appears."""
    calls = []

    def embed(s):
        calls.append(s)
        return VEC[s]

    pairs = [("a cat", "a dog"), ("a cat", "stocks fell"), ("a dog", "a cat")]
    r = sbert(pairs, embed)
    cos = lambda u, v: sum(a * b for a, b in zip(u, v)) / math.sqrt(
        sum(a * a for a in u) * sum(b * b for b in v))
    assert r["scores"] == pytest.approx([cos(VEC[a], VEC[b]) for a, b in pairs], rel=1e-14)
    assert r["embed_calls"] == 3 and sorted(calls) == sorted(VEC)
    assert r["cross_encoder_calls"] == 3


def test_sbert_edge():
    """A sentence scored against itself has cosine 1."""
    r = sbert([("a dog", "a dog")], lambda s: VEC[s])
    assert r["scores"] == pytest.approx([1.0], rel=1e-15)
    assert r["embed_calls"] == 1
