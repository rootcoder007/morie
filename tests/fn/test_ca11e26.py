"""Tests for ca11e26.ca_chapter_11_equation_26."""

from morie.fn import _array_core as np

from morie.fn.ca11e26 import ca_chapter_11_equation_26


def test_ca11e26_basic():
    """Test basic functionality."""
    d = 3
    result = ca_chapter_11_equation_26(d)
    assert isinstance(result, dict)
    assert "value" in result
def test_ca11e26_edge():
    """Test edge cases."""
    d = 3
    result = ca_chapter_11_equation_26(d)
    assert isinstance(result, dict)
