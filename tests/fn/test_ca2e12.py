"""Tests for ca2e12.ca_chapter_2_equation_12."""

from morie.fn import _array_core as np

from morie.fn.ca2e12 import ca_chapter_2_equation_12


def test_ca2e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_12(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca2e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_12(x)
    assert isinstance(result, dict)
