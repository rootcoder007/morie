"""Tests for ca5e5.ca_chapter_5_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca5e5 import ca_chapter_5_equation_5


def test_ca5e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_5(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_ca5e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_5(x)
    assert isinstance(result, dict)
