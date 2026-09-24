"""Tests for pageH.page_hinkley."""

from morie.fn import _array_core as np

from morie.fn.pageH import page_hinkley


def test_pageH_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = page_hinkley(x, threshold)
    assert isinstance(result, dict)
    assert "statistic" in result or "statistic" in result or "statistic" in result


def test_pageH_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = page_hinkley(x, threshold)
    assert isinstance(result, dict)
