"""Tests for morie.fn.ngram."""

from morie.fn import _array_core as np

from morie.fn.ngram import ngram_freq


def test_ngram_smoke():
    rng = np.random.default_rng(42)
    result = ngram_freq(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ngram import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
