"""Tests for fastxt -- Bojanowski, Grave, Joulin & Mikolov (2017).

Replaces a generated test that called a stub returning mean(corpus).
Full anchor: ledger/wave3/anchor_embed.py.
"""

import math

import pytest

from morie.fn.fastxt import fasttext, subwords, word_vector

DOCS = [["cat", "sat", "mat"] * 8, ["dog", "ran", "far"] * 8]


def cosine(a, b):
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def test_the_papers_where_example():
    """Sec. 3.2 prints this decomposition; it must match exactly."""
    assert subwords("where", 3, 3) == ["<wh", "whe", "her", "ere",
                                       "re>", "<where>"]


def test_boundaries_separate_prefix_from_interior():
    """'her' is interior to both words, but '<he' and '<her>' are not."""
    assert "her" in subwords("where", 3, 3)
    assert "her" in subwords("her", 3, 3)
    assert "<he" in subwords("her", 3, 3)
    assert "<he" not in subwords("where", 3, 3)
    assert "<her>" in subwords("her", 3, 3)
    assert "<her>" not in subwords("where", 3, 3)
    # dropping the boundaries merges them
    assert subwords("where", 3, 3, boundary=False) == [
        "whe", "her", "ere", "where"]


def test_ngram_range_is_inclusive():
    lens = set(len(g) for g in
               subwords("abcdefgh", 3, 6, whole_word=False))
    assert lens == {3, 4, 5, 6}
    with pytest.raises(ValueError):
        subwords("x", 4, 2)
    with pytest.raises(ValueError):
        subwords("x", 0, 3)


def test_training_reduces_the_loss():
    f = fasttext(DOCS, dim=12, epochs=4, seed=1)
    assert f["final_loss"] < f["loss_history"][0]


def test_out_of_vocabulary_words_get_vectors():
    """The reason the model exists: unseen words share n-grams."""
    f = fasttext(DOCS, dim=12, epochs=4, seed=1)
    v = f["oov"]("catt")
    assert any(abs(t) > 1e-12 for t in v)
    assert (cosine(v, f["vectors"][f["index"]["cat"]])
            > cosine(v, f["vectors"][f["index"]["far"]]))


def test_a_word_sharing_nothing_gets_exact_zero():
    f = fasttext(DOCS, dim=12, epochs=2, seed=1)
    v, hits = word_vector("zzzzzz", f["Z"], f["ngram_index"])
    assert hits == 0
    assert all(t == 0.0 for t in v)


def test_argument_checks():
    with pytest.raises(ValueError):
        fasttext([["only"]], dim=4)
    with pytest.raises(ValueError):
        fasttext(DOCS, dim=0)
    with pytest.raises(ValueError):
        fasttext(None)
