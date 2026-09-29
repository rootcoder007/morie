"""Tests for morie.fn.ngram."""

from morie.fn.ngram import ngram_freq


def test_ngram_smoke():
    result = ngram_freq(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ngram import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_ngram_counts_recomputed():
    text = "the cat sat on the mat the cat ran"
    r = ngram_freq(text, n=2, top_k=3)
    toks = text.split()
    grams = [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
    assert r.extra["total_ngrams"] == len(grams)
    assert r.extra["unique_ngrams"] == len(set(grams))
    assert r.extra["ngrams"][0] == ("the cat", 2)
