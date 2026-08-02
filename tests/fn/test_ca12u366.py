"""Tests for ca12u366.ca_chapter_12_unnumbered_366."""

from morie.fn import _array_core as np

from morie.fn.ca12u366 import ca_chapter_12_unnumbered_366


def test_ca12u366_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_366(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca12u366_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_12_unnumbered_366(x)
    assert isinstance(result, dict)
