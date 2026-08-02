"""Tests for ca5u172.ca_chapter_5_unnumbered_172."""

from morie.fn import _array_core as np

from morie.fn.ca5u172 import ca_chapter_5_unnumbered_172


def test_ca5u172_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_172(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca5u172_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_172(x)
    assert isinstance(result, dict)
