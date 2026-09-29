"""Tests for morie.fn.wrdcl."""

from morie.fn.wrdcl import word_cloud_data


def test_wrdcl_smoke():
    result = word_cloud_data(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.wrdcl import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_word_counts_after_stopwords():
    r = word_cloud_data("The cat and the dog; the cat ran. A dog!", top_k=2)
    assert r.extra["words"] == [("cat", 2), ("dog", 2)]
    assert r.extra["total_words"] == 5
    assert r.extra["unique_words"] == 3
    assert r.extra["normalized"][0] == ("cat", 2, 1.0)
