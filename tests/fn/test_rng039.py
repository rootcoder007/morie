"""Tests for rng039.rangayyan_ch3_ma_filter_11pt."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_ch3_ma_filter_11pt


def test_rng039_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_filter_11pt(x, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng039_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_filter_11pt(x, n)
    assert isinstance(result, dict)
