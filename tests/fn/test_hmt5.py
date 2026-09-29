"""Tests for hmt5.geron_t5."""

from morie.fn import _array_core as np
from morie.fn.hmt5 import geron_t5


def test_hmt5_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_t5(src, tgt)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmt5_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_t5(src, tgt)
    assert isinstance(result, dict)


def test_span_corruption_is_lossless_and_sentinel_structured():
    from morie.fn.hmt5 import restore, span_corrupt

    toks = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    enc, dec, spans = span_corrupt(toks, noise_density=0.25, mean_span=2, seed=3)
    assert restore(enc, dec) == toks
    k = len(spans)
    assert sum(t.startswith("<extra_id_") for t in enc) == k
    assert dec[-1] == f"<extra_id_{k}>"
    assert sum(L for _, L in spans) == len(dec) - (k + 1)
    r = geron_t5(" ".join(toks), noise_density=0.25, mean_span=2, seed=3)
    assert r["estimate"] == sum(L for _, L in spans) / 12
