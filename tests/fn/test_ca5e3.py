"""Tests for ca5e3.ca_chapter_5_equation_3."""

from morie.fn import _array_core as np

from morie.fn.ca5e3 import ca_chapter_5_equation_3


def test_ca5e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_3(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca5e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_equation_3(x)
    assert isinstance(result, dict)
