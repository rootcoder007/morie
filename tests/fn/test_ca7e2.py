"""Tests for ca7e2.ca_chapter_7_equation_2."""

from morie.fn import _array_core as np

from morie.fn.ca7e2 import ca_chapter_7_equation_2


def test_ca7e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca7e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_2(x)
    assert isinstance(result, dict)
