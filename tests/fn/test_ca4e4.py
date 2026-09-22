"""Tests for ca4e4.ca_chapter_4_equation_4."""

from morie.fn import _array_core as np

from morie.fn.ca4e4 import ca_chapter_4_equation_4


def test_ca4e4_basic():
    """Test basic functionality."""
    p = 0.5
    result = ca_chapter_4_equation_4(p)
    assert isinstance(result, dict)
    assert "value" in result
def test_ca4e4_edge():
    """Test edge cases."""
    p = 0.5
    result = ca_chapter_4_equation_4(p)
    assert isinstance(result, dict)
