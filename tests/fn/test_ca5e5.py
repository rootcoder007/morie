"""Tests for ca5e5.ca_chapter_5_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca5e5 import ca_chapter_5_equation_5


def test_ca5e5_basic():
    """Test basic functionality."""
    b = 0.5
    se = 0.1
    result = ca_chapter_5_equation_5(b, se)
    assert result.lower == b - 1.96 * se
    assert result.upper == b + 1.96 * se
    assert "value" in result
    assert result.value == b - 1.96 * se


def test_ca5e5_edge():
    """Test edge cases."""
    b = -0.3
    se = 0.2
    result = ca_chapter_5_equation_5(b, se)
    assert result.lower == b - 1.96 * se
    assert result.upper == b + 1.96 * se
